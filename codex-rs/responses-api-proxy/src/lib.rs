use std::fs::File;
use std::fs::{self};
use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use clap::Parser;
use clap::ValueEnum;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::AUTHORIZATION;
use reqwest::header::HOST;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderName;
use reqwest::header::HeaderValue;
use serde::Serialize;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Request;
use tiny_http::Response;
use tiny_http::Server;
use tiny_http::StatusCode;

mod dump;
mod read_api_key;
mod translate;
use dump::ExchangeDumper;
use read_api_key::read_auth_header_from_stdin;
use translate::MappedRequestBody;

/// CLI arguments for the proxy.
#[derive(Debug, Clone, Parser)]
#[command(name = "responses-api-proxy", about = "Minimal OpenAI responses proxy")]
pub struct Args {
    /// Port to listen on. If not set, an ephemeral port is used.
    #[arg(long)]
    pub port: Option<u16>,

    /// Path to a JSON file to write startup info (single line). Includes {"port": <u16>}.
    #[arg(long, value_name = "FILE")]
    pub server_info: Option<PathBuf>,

    /// Enable HTTP shutdown endpoint at GET /shutdown
    #[arg(long)]
    pub http_shutdown: bool,

    /// Absolute URL the proxy should forward requests to (defaults to OpenAI).
    #[arg(long, default_value = "https://api.openai.com/v1/responses")]
    pub upstream_url: String,

    /// Upstream wire API format used at --upstream-url.
    #[arg(long, default_value = "responses")]
    pub upstream_wire_api: UpstreamWireApi,

    /// Directory where request/response dumps should be written as JSON.
    #[arg(long, value_name = "DIR")]
    pub dump_dir: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum UpstreamWireApi {
    Responses,
    #[value(name = "chat-completions")]
    ChatCompletions,
}

#[derive(Serialize)]
struct ServerInfo {
    port: u16,
    pid: u32,
}

struct ForwardConfig {
    upstream_url: Url,
    host_header: HeaderValue,
    upstream_wire_api: UpstreamWireApi,
}

/// Entry point for the library main, for parity with other crates.
pub fn run_main(args: Args) -> Result<()> {
    let auth_header = read_auth_header_from_stdin()?;

    let upstream_url = Url::parse(&args.upstream_url).context("parsing --upstream-url")?;
    let host = match (upstream_url.host_str(), upstream_url.port()) {
        (Some(host), Some(port)) => format!("{host}:{port}"),
        (Some(host), None) => host.to_string(),
        _ => return Err(anyhow!("upstream URL must include a host")),
    };
    let host_header =
        HeaderValue::from_str(&host).context("constructing Host header from upstream URL")?;

    let forward_config = Arc::new(ForwardConfig {
        upstream_url,
        host_header,
        upstream_wire_api: args.upstream_wire_api,
    });
    let dump_dir = args
        .dump_dir
        .map(ExchangeDumper::new)
        .transpose()
        .context("creating --dump-dir")?
        .map(Arc::new);

    let (listener, bound_addr) = bind_listener(args.port)?;
    if let Some(path) = args.server_info.as_ref() {
        write_server_info(path, bound_addr.port())?;
    }
    let server = Server::from_listener(listener, None)
        .map_err(|err| anyhow!("creating HTTP server: {err}"))?;
    let client = Arc::new(
        Client::builder()
            // Disable reqwest's 30s default so long-lived response streams keep flowing.
            .timeout(None::<Duration>)
            .build()
            .context("building reqwest client")?,
    );

    eprintln!("responses-api-proxy listening on {bound_addr}");

    let http_shutdown = args.http_shutdown;
    for request in server.incoming_requests() {
        let client = client.clone();
        let forward_config = forward_config.clone();
        let dump_dir = dump_dir.clone();
        std::thread::spawn(move || {
            if http_shutdown && request.method() == &Method::Get && request.url() == "/shutdown" {
                let _ = request.respond(Response::new_empty(StatusCode(200)));
                std::process::exit(0);
            }

            if let Err(e) = forward_request(
                &client,
                auth_header,
                &forward_config,
                dump_dir.as_deref(),
                request,
            ) {
                eprintln!("forwarding error: {e}");
            }
        });
    }

    Err(anyhow!("server stopped unexpectedly"))
}

fn bind_listener(port: Option<u16>) -> Result<(TcpListener, SocketAddr)> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port.unwrap_or(0)));
    let listener = TcpListener::bind(addr).with_context(|| format!("failed to bind {addr}"))?;
    let bound = listener.local_addr().context("failed to read local_addr")?;
    Ok((listener, bound))
}

fn write_server_info(path: &Path, port: u16) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let info = ServerInfo {
        port,
        pid: std::process::id(),
    };
    let mut data = serde_json::to_string(&info)?;
    data.push('\n');
    let mut f = File::create(path)?;
    f.write_all(data.as_bytes())?;
    Ok(())
}

fn forward_request(
    client: &Client,
    auth_header: &'static str,
    config: &ForwardConfig,
    dump_dir: Option<&ExchangeDumper>,
    mut req: Request,
) -> Result<()> {
    // Only allow POST /v1/responses exactly, no query string.
    let method = req.method().clone();
    let url_path = req.url().to_string();
    let allow = method == Method::Post && url_path == "/v1/responses";

    if !allow {
        let resp = Response::new_empty(StatusCode(403));
        let _ = req.respond(resp);
        return Ok(());
    }

    // Read request body
    let mut body = Vec::new();
    let reader = req.as_reader();
    reader.read_to_end(&mut body)?;

    let exchange_dump = dump_dir.and_then(|dump_dir| {
        dump_dir
            .dump_request(&method, &url_path, req.headers(), &body)
            .map_err(|err| {
                eprintln!("responses-api-proxy failed to dump request: {err}");
                err
            })
            .ok()
    });

    let mapped_request = match config.upstream_wire_api {
        UpstreamWireApi::Responses => {
            let stream = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|request| request.get("stream").and_then(serde_json::Value::as_bool))
                .unwrap_or(false);
            MappedRequestBody { body, stream }
        }
        UpstreamWireApi::ChatCompletions => {
            translate::responses_to_chat_completions_request(&body)?
        }
    };

    // Build headers for upstream, forwarding everything from the incoming
    // request except Authorization (we replace it below).
    let mut headers = HeaderMap::new();
    for header in req.headers() {
        let name_ascii = header.field.as_str();
        let lower = name_ascii.to_ascii_lowercase();
        if lower.as_str() == "authorization" || lower.as_str() == "host" {
            continue;
        }

        let header_name = match HeaderName::from_bytes(lower.as_bytes()) {
            Ok(name) => name,
            Err(_) => continue,
        };
        if let Ok(value) = HeaderValue::from_bytes(header.value.as_bytes()) {
            headers.append(header_name, value);
        }
    }

    // As part of our effort to to keep `auth_header` secret, we use a
    // combination of `from_static()` and `set_sensitive(true)`.
    let mut auth_header_value = HeaderValue::from_static(auth_header);
    auth_header_value.set_sensitive(true);
    headers.insert(AUTHORIZATION, auth_header_value);

    headers.insert(HOST, config.host_header.clone());

    let upstream_resp = client
        .post(config.upstream_url.clone())
        .headers(headers)
        .body(mapped_request.body)
        .send()
        .context("forwarding request to upstream")?;

    let status = upstream_resp.status();
    let upstream_headers = upstream_resp.headers().clone();

    let mut upstream_body = Vec::new();
    let mut upstream_resp = upstream_resp;
    upstream_resp
        .read_to_end(&mut upstream_body)
        .context("reading upstream response body")?;

    let (response_body_bytes, force_content_type) = match config.upstream_wire_api {
        UpstreamWireApi::Responses => (upstream_body, None),
        UpstreamWireApi::ChatCompletions => {
            if status.is_success() {
                if mapped_request.stream {
                    (
                        translate::chat_completions_sse_to_responses_sse(&upstream_body)?,
                        Some("text/event-stream"),
                    )
                } else {
                    (
                        translate::chat_completions_json_to_responses_json(&upstream_body)?,
                        Some("application/json"),
                    )
                }
            } else {
                (upstream_body, None)
            }
        }
    };

    // We have to create an adapter between a `reqwest::blocking::Response`
    // and a `tiny_http::Response`. Fortunately, `reqwest::blocking::Response`
    // implements `Read`, so we can use it directly as the body of the
    // `tiny_http::Response`.
    let mut response_headers = Vec::new();
    let mut content_type_overridden = false;
    for (name, value) in upstream_headers.iter() {
        // Skip headers that tiny_http manages itself.
        if matches!(
            name.as_str(),
            "content-length" | "transfer-encoding" | "connection" | "trailer" | "upgrade"
        ) {
            continue;
        }

        if let Some(force_content_type) = force_content_type
            && name.as_str().eq_ignore_ascii_case("content-type")
        {
            if let Ok(header) =
                Header::from_bytes(b"content-type".as_slice(), force_content_type.as_bytes())
            {
                response_headers.push(header);
                content_type_overridden = true;
            }
            continue;
        }

        if let Ok(header) = Header::from_bytes(name.as_str().as_bytes(), value.as_bytes()) {
            response_headers.push(header);
        }
    }

    if let Some(force_content_type) = force_content_type
        && !content_type_overridden
        && let Ok(header) =
            Header::from_bytes(b"content-type".as_slice(), force_content_type.as_bytes())
    {
        response_headers.push(header);
    }

    let content_length = Some(response_body_bytes.len());
    let response_body: Box<dyn Read + Send> = if let Some(exchange_dump) = exchange_dump {
        let mut dump_headers = reqwest::header::HeaderMap::new();
        for header in &response_headers {
            if let Ok(name) =
                reqwest::header::HeaderName::from_bytes(header.field.as_str().as_bytes())
                && let Ok(value) = reqwest::header::HeaderValue::from_bytes(header.value.as_bytes())
            {
                dump_headers.append(name, value);
            }
        }
        Box::new(exchange_dump.tee_response_body(
            status.as_u16(),
            &dump_headers,
            Cursor::new(response_body_bytes),
        ))
    } else {
        Box::new(Cursor::new(response_body_bytes))
    };

    let response = Response::new(
        StatusCode(status.as_u16()),
        response_headers,
        response_body,
        content_length,
        None,
    );

    let _ = req.respond(response);
    Ok(())
}
