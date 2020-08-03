import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext

# create a connection to database
def get_db():
    if  'db' not in g:
        g.db = sqlite3.connect( # establishes a connection to the file pointed at by the DATABASE configuration key. 
            current_app.config['DATABASE'],# is another special object that points to the Flask application handling the request. 
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    
    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

#Add the Python functions that will run these SQL commands 

def init_db():
    db = get_db()#connect data

    with current_app.open_resource('schema.sql') as f:# opens a file relative to the flaskr package
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
   """Clear the existing data and create new tables."""
   init_db()
   click.echo('Initialized the database.')


def init_app(app):
    """Register database functions with the Flask app. This is called by
    the application factory.
    """
    app.teardown_appcontext(close_db)# tells Flask to call that function when cleaning up after returning the response.
    app.cli.add_command(init_db_command)# adds a new command that can be called with the flask command.

