from flask import Flask, render_template
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
from SongFingerPrinting import SongFingerPrinting as fp

app = Flask(__name__)
ask = Ask(app, '/')

songfp = fp()
songfp.example_load()


@app.route('/')
def homepage():
    return "Hello my peeps"

@ask.launch
def start_skill():
    msg = "Would you like me to identify the song excerpt?"
    return question(msg)

@ask.intent("YesIntent")
def rate_song():
    song = songfp.make_excerpt(8)
    out = songfp.match_song(song)
    print(out)
    return statement(out)

@ask.intent("NoIntent")
def no_intent():
    msg = "Ok, thanks. Have a nice day lol."
    return statement(msg)

if __name__ == '__main__':
    app.run(debug=True)