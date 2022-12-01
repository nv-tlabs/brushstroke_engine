
## Neural Brushstroke Engine Web UI

Web UI and python server that allow drawing a stroke
in a web browser, with the stroke rendered patch
by patch on the server side and all patches
are sent back to the client and rendered.

Current implementation contains only a stub for a
back end stroke renderer that draws a red frame
around the target patch.

### Code Organization

* `ui/js` - javascript source code
* `ui/static` - files served to the front end
* `ui/templates` - jinja HTML templates
* `ui/` - core python code

### Running

To run the server, compile javscript:
```bash
cd forger/ui
make clean
make
cd ../..
python -m forger.ui.run --port=8000 --log_level=10 \
--gan_checkpoint="path/to/gan_checkpoint" \
--encoder_checkpoint="path/to/encoder_checkpoint"
```

Then, navigate to [http://localhost:8000/](http://localhost:8000/) 
(only tested in Chrome and Safari).

Note that the URL comes with two notable parameters:
* `demo` mode automatically sets best stitching and background settings, to use go to: [http://localhost:8000/?demo](http://localhost:8000/?demo) 
* `canvas` allows setting drawing canvas width, like so: [http://localhost:8000/?demo&canvas=2000](http://localhost:8000/?demo&canvas=2000)  (use larger values for higher res screens)

### General Architecture

The client side uses two HTML5 Canvases:
* **background canvas:** used for drawing plain stroke
based on user mouse interaction 
  (this is the conditioning input for our model)
* **foreground canvas:** used to draw the final stroke
received from the server
  
The client can send and receive JSON or binary messages
through a web socket. Whenever the user-drawn stroke
exits a patch of certain `patch_width` centered at
the stroke start, the client encodes both the 
stroke canvas and the rendered stroke as binary and
sends a request to the server. When the server
responds with the updated foreground pixels, the
client updates the **foreground canvas**. In the future
the client should be able to send additional
settings, like the tool type and color.