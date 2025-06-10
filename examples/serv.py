# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi==0.115.12",
#     "marimo",
#     "mohtml==0.1.10",
#     "requests==2.32.3",
#     "uvicorn==0.34.3",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Server Setup""")
    return


@app.cell
def _():
    import marimo as mo
    import threading
    import asyncio
    import time
    import uvicorn
    return asyncio, mo, threading, time, uvicorn


@app.cell
def _(mo):
    get_server_thread, set_server_thread = mo.state(None)
    get_current_server, set_current_server = mo.state(None)
    return (
        get_current_server,
        get_server_thread,
        set_current_server,
        set_server_thread,
    )


@app.cell
def _(
    PORT,
    StoppableServer,
    app,
    asyncio,
    get_current_server,
    get_server_thread,
    set_current_server,
    set_server_thread,
    threading,
    time,
):
    # Stop existing server
    if get_server_thread() and get_server_thread().is_alive():
        print("Stopping current server...")
        if get_current_server():
            server = get_current_server()
            server.stop()
            set_current_server(server)

        # Wait for thread to finish
        thread = get_server_thread()
        thread.join(timeout=2)
        time.sleep(0.5)  # Give port time to be released

    # Create new server
    set_current_server(StoppableServer(app, host="0.0.0.0", port=PORT))


    def run_server():
        asyncio.run(get_current_server().serve())


    current_server_thread = threading.Thread(target=run_server, daemon=True)
    current_server_thread.start()
    set_server_thread(current_server_thread)

    print(f"Server started on thread {current_server_thread.ident}")
    return


@app.cell
def _(uvicorn):
    class StoppableServer:
        def __init__(self, app, host="0.0.0.0", port=8000):
            self.config = uvicorn.Config(app, host=host, port=port)
            self.server = uvicorn.Server(self.config)

        async def serve(self):
            await self.server.serve()

        def stop(self):
            if self.server:
                self.server.should_exit = True
                # Also force close any existing servers
                if hasattr(self.server, "servers"):
                    for server in self.server.servers:
                        server.close()
    return (StoppableServer,)


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Developer Experience live updates!""")
    return


@app.cell
def _(FastAPI, HTMLResponse, JSONResponse, div, p):
    app = FastAPI()


    @app.get("/", response_class=HTMLResponse)
    def read_root():
        return str(div(p("hello", klass="text-2xl text-center text-blue-500")))


    @app.get("/api", response_class=JSONResponse)
    def read_root():
        return {"alive": "yes!!"}
    return (app,)


@app.cell
def _(PORT, div, rq):
    div(rq.get(f"http://localhost:{PORT}/").text)
    return


@app.cell
def _(PORT, rq):
    rq.get(f"http://localhost:{PORT}/api").json()
    return


@app.cell
def _():
    from mohtml import div, p, head, body, tailwind_css, html
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, JSONResponse

    tailwind_css()
    return FastAPI, HTMLResponse, JSONResponse, div, p


@app.cell
def _(get_server_thread):
    import requests as rq

    get_server_thread()
    return (rq,)


@app.cell
def _():
    PORT = 12345
    return (PORT,)


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
