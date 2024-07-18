import eventlet
from eventlet import wsgi
from engineio.middleware import WSGIApp
from app import app, socketio  # Import both app and socketio

eventlet.monkey_patch() 

application = WSGIApp(socketio, app)

# if __name__ == '__main__':
#     wsgi.server(eventlet.listen(('', 5000)), application)  # Pass the wrapped application to the server
