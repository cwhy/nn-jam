import trio
from pynng import Pair0
from trio_websocket import open_websocket_url, serve_websocket, ConnectionClosed

port_internal = 54321
port_external = 12345
socket = Pair0(listen=f'tcp://127.0.0.1:{port_internal}')


async def echo_server(request):
    ws = await request.accept()
    while True:
        try:
            message = await socket.arecv()
            await ws.send_message(message)
            # message = await ws.get_message()
            # print('Received message: %s' % message)
        except ConnectionClosed:
            break


async def main():
    await serve_websocket(echo_server, '127.0.0.1', port_external, ssl_context=None)


trio.run(main)
