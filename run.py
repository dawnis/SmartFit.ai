#!/usr/bin/env python
from flaskapp import app, views
views.my_load_model()
app.run(host='0.0.0.0', debug = True)
