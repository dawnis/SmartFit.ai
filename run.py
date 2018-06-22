#!/usr/bin/env python
from flaskapp import app, views
views.my_load_model()
app.run(debug = True)
