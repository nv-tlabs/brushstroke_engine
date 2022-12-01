// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
var nvidia = nvidia || {};

/**
 * Keeps track of a bounding box around a set of points.
 */
nvidia.BoundingBox = function() {
  this.min_x = null;
  this.min_y = null;
  this.max_x = null;
  this.max_y = null;
};

nvidia.BoundingBox.prototype.addPoint = function(x, y) {
  if (this.min_x === null) {
    this.min_x = x;
    this.max_x = x;
    this.min_y = y;
    this.max_y = y;
  } else {
    this.min_x = Math.min(this.min_x, x);
    this.max_x = Math.max(this.max_x, x);
    this.min_y = Math.min(this.min_y, y);
    this.max_y = Math.max(this.max_y, y);
  }
};

nvidia.BoundingBox.prototype.getWidth = function() {
  if (this.min_x === null) {
    return 0;
  }
  return Math.ceil(this.max_x - this.min_x) + 1;
};

nvidia.BoundingBox.prototype.getHeight = function() {
  if (this.min_y === null) {
    return 0;
  }
  return Math.ceil(this.max_y - this.min_y) + 1;
};


nvidia.BoundingBox.prototype.maxDim = function() {
  if (this.min_x === null) {
    return 0;
  }

  return Math.max(this.getWidth(), this.getHeight());
};

nvidia.BoundingBox.prototype.getCenter = function() {
  if (this.min_x === null) {
    nvidia.util.timed_log('Requesting center of uninitialized BoundingBox', 'ERROR');
    return [0, 0];
  }
  return [ Math.round(this.min_x + (this.max_x - this.min_x + 1) / 2),
           Math.round(this.min_y + (this.max_y - this.min_y + 1) / 2) ];
};


//TODO: update description

/**
 * Allows using a mouse to draw on an HTML5 canvas. Each user stroke
 * begins at the center of a patch of width patch_width. As soon as the
 * drawn stroke exits this patch, the controller Will issue a
 * "window_complete" event from the canvas with the location of the
 * patch center. The center of the drawing window will then be reset.
 * Thus, the controller issues events for overlapping drawing patches
 * that may need to be drawn.
 *
 * @param {String} canvas_selector jquery selector for the darwing canvas
 * @param {Number} patch_width width of the drawing window; should be smaller
 *    than the server-side generator resolution
 */
nvidia.DrawingController = function(canvas_selector, patch_width) {
  this.canvas = $(canvas_selector);
  this.ctx = this.canvas[0].getContext('2d');
  this.patch_width = patch_width;
  this.thickness = 10;
  this.dirty_window = new nvidia.BoundingBox();

  // Previously drawn point
  this.touchManager = new nvidia.TouchManager();
  this.x = 0;
  this.y = 0;
  this.trigger_force_event = false;

  // Center of the current drawing window
  //this.window_x = 0;
  //this.window_y = 0;

  this.init();
};

nvidia.DrawingController.prototype.resetDirtyWindow = function() {
  this.dirty_window = new nvidia.BoundingBox();
};

nvidia.DrawingController.prototype.clearCanvas = function() {
  this.ctx.clearRect(0, 0, this.canvas[0].width, this.canvas[0].height);
  this.resetDirtyWindow();
};

/**
 * Initializes canvas events.
 */
nvidia.DrawingController.prototype.init = function() {
  let dcontroller = this;
  $(this.canvas).on('mousedown', e => { dcontroller.onMouseDown(e); });
  $(this.canvas).on('mouseup', e => { dcontroller.onMouseUp(e); });
  $(this.canvas).on('mouseleave', e => { dcontroller.onMouseUp(e); });

  // Mobile events
  $(this.canvas).on('touchstart', e => { dcontroller.onTouchStart(e); });
  $(this.canvas).on('touchend touchcancel', e => { dcontroller.onTouchEnd(e); });
};

/**
 * Obtains the coordinates within the HTML5 canvas (for drawing) frorm
 * the JS event object that issued on the canvas. This is necessary
 * because canvas resolution can differ from the display width of the
 * canvas element.
 */
nvidia.DrawingController.prototype.canvasCoordsFromEvent = function(e) {
  let x_disp = e.pageX - this.canvas.offset().left;
  let y_disp = e.pageY - this.canvas.offset().top;
  let x = x_disp / this.canvas.width() * this.canvas[0].width;
  let y = y_disp / this.canvas.height() * this.canvas[0].height;
  return [x, y]
};

nvidia.DrawingController.prototype.startStroke = function(e) {
  let coords = this.canvasCoordsFromEvent(e);
  this.x = coords[0];
  this.y = coords[1];
  this.resetDirtyWindow();
  this.dirty_window.addPoint(Math.floor(this.x), Math.floor(this.y));
};

nvidia.DrawingController.prototype.onMouseDown = function(e) {
  this.startStroke(e);
  let dcontroller = this;
  $(this.canvas).on('mousemove', e => { dcontroller.onMouseMove(e);});
};

nvidia.DrawingController.prototype.onTouchStart = function(e) {
  e.preventDefault();
  e.stopPropagation();
  this.touchManager.registerTouches(e);

  if (this.touchManager.active_touch !== null) {
    nvidia.util.ensurePageX(e);
    this.startStroke(e);

    let dcontroller = this;
    $(this.canvas).on('touchmove', e => { dcontroller.onTouchMove(e);});
  }
};

nvidia.DrawingController.prototype.onMouseUp = function(e) {
  $(this.canvas).off('mousemove');

  if (this.dirty_window.maxDim() > 0) {
    this.canvas.trigger(
      "window_complete",
      { 'dirty': this.dirty_window, 'stroke_end': true});
    this.resetDirtyWindow();
  }
};

nvidia.DrawingController.prototype.onTouchEnd = function(e) {
  nvidia.util.ensurePageX(e);

  var tids = nvidia.TouchManager.getTouchIds(e, e.changedTouches);
  var externalTouch = true;
  for (var i = 0; i < tids.length; ++i) {
    if (this.touchManager.isTouchRegistered(tids[i])) {
      externalTouch = false;
    }
  }

  if (externalTouch) {  // Could be touch end for anything
    return;
  }

  e.preventDefault();
  e.stopPropagation();

  this.touchManager.unregisterTouches(e);

  if (this.touchManager.active_touch === null) {
    $(this.canvas).off('touchmove');

    if (this.dirty_window.maxDim() > 0) {
      this.canvas.trigger("window_complete",
                          { 'dirty': this.dirty_window, 'stroke_end': true });
      this.resetDirtyWindow();
    }
  }
};

nvidia.DrawingController.prototype.onMouseMove = function(e) {
  // Draw next segment
  let dcontroller = this;
  window.requestAnimationFrame(function() { dcontroller.handleDrawAnimationFrame(e) });
};

nvidia.DrawingController.prototype.handleDrawAnimationFrame = function(e) {
  let coords = this.canvasCoordsFromEvent(e);
  this.drawNextSegmentWithCircle(coords[0], coords[1], true);
  this.x = coords[0];
  this.y = coords[1];
};

nvidia.DrawingController.prototype.onTouchMove = function(e) {
  if (this.touchManager.updateTouches(e)) {
    let pos_e = this.touchManager.getActiveTouchPosition();
    if (pos_e !== null) {
      let dcontroller = this;
      window.requestAnimationFrame(function() { dcontroller.handleDrawAnimationFrame(pos_e) });
    }

    if (this.trigger_force_event) {
      let info = this.touchManager.getActiveTouchInfo();
      if (info !== null && info['force'] !== null) {
        this.canvas.trigger("force", {'force': info.force});
      }
    }

    e.preventDefault();
    e.stopPropagation();
  }
};

/**
 * Draws a segment from current x,y to the next x,y (provided).
 */
nvidia.DrawingController.prototype.drawNextSegment = function(x, y) {
  this.drawNextSegmentWithCircle(x, y);
};

nvidia.DrawingController.prototype.drawNextSegmentWithCircle = function(x, y, with_events) {
  let delta_x = x - this.x;
  let delta_y = y - this.y;
  let distance = Math.sqrt(Math.pow(delta_x, 2) + Math.pow(delta_y, 2));
  let ncircles = Math.max(2, Math.ceil(distance / this.thickness));
  this.ctx.beginPath();
  let dirty_windows = [];

  for (let i = 0; i < ncircles; ++i) {
    let xx = Math.round(this.x + delta_x * i / (ncircles - 1));
    let yy = Math.round(this.y + delta_y * i / (ncircles - 1));
    this.ctx.moveTo(xx, yy);
    this.ctx.arc(xx, yy, this.thickness, 0, Math.PI * 2);

    if (with_events) {
      this.dirty_window.addPoint(Math.floor(xx), Math.floor(yy));
      if (this.dirty_window.maxDim() >= this.patch_width) {
        dirty_windows.push(this.dirty_window);
        this.resetDirtyWindow();
      }
    }
  }

  this.ctx.strokeStyle = 'black';
  this.ctx.fillStyle = 'black';
  this.ctx.fill();

  for (let i = 0; i < dirty_windows.length; ++i) {
    this.canvas.trigger("window_complete", { 'dirty': dirty_windows[i] });
  }
};

nvidia.DrawingController.prototype.drawNextSegmentWithPath = function(x, y) {
  this.ctx.beginPath();
  this.ctx.strokeStyle = 'black';
  this.ctx.lineWidth = this.thickness;
  this.ctx.moveTo(this.x, this.y);
  this.ctx.lineTo(x, y);
  this.ctx.stroke();
  this.ctx.closePath();
};
// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
var nvidia = nvidia || {};

/**
 * Main controller class responsible for managing the entire application.
 *
 * @param {String} bg_canvas_sel background canvas selector
 * @param {String} fg_canvas_sel foreground canvas selector
 * @param {Number} patch_width width of the patch to send to server
 *                             (will be updated when server responds)
 */
nvidia.Controller = function(bg_canvas_sel, fg_canvas_sel, patch_width) {
  this.patch_width = patch_width;
  this.crop_margin = 0;
  this.drawController = new nvidia.DrawingController(
    bg_canvas_sel, Math.floor(patch_width / 2));

  this.bg_canvas_sel = bg_canvas_sel;
  this.fg_canvas_sel = fg_canvas_sel;
  this.render_canvas = $(fg_canvas_sel);
  this.ctx = this.render_canvas[0].getContext("2d");
  this.ctx.imageSmoothingEnabled = true;

  this.ws = null;              // web socket
  this.ws_ready = false;
  this.feature_blending_mode = 0;  // 0 - disabled, 1 - output res, 2 - res/2, etc
  this.server_side_geometry = false;
  this.auto_new_layer = false;
  this.undoStack = [];
  this.redoStack = [];
  this.maxUndos = 10;

  this.current_seed = -1;
  this.current_brush_id = '';
  this.demo_mode = false;
  this.demo_seed_id = 0;
  this.demo_seeds = [];

  // Layers
  this.baked_bg_canvas_sel = null;
  this.baked_fg_canvas_sel = null;

  // Demo of force style
  this.forcedemo = null;

  this.init();
};

nvidia.Controller.prototype.downloadAll = function(prefix) {
  if (prefix) {
    prefix = prefix + '_';
  } else {
    prefix = '';
  }
  prefix = 'forger_' + prefix + 'seed' + this.current_brush_id;

  nvidia.util.downloadURL(prefix + '_fake.png', $(this.fg_canvas_sel)[0].toDataURL('image/png'));
  nvidia.util.downloadURL(prefix + '_stroke.png', $(this.bg_canvas_sel)[0].toDataURL('image/png'));

  if (this.baked_bg_canvas_sel) {
    nvidia.util.downloadURL(prefix + '_stroke_baked.png', $(this.baked_bg_canvas_sel)[0].toDataURL('image/png'));
  }
  if (this.baked_fg_canvas_sel) {
    nvidia.util.downloadURL(prefix + '_fake_baked.png', $(this.baked_fg_canvas_sel)[0].toDataURL('image/png'));
  }
};

nvidia.Controller.prototype.downloadStroke = function(prefix) {
  if (this.baked_bg_canvas_sel) {
    nvidia.util.downloadURL(prefix + '_stroke_baked.png', $(this.baked_bg_canvas_sel)[0].toDataURL('image/png'));
  }
};

nvidia.Controller.prototype.setFeatureBlending = function(mode) {
  const requires_new_canvas = this.feature_blending_mode != mode || mode > 0;
  this.feature_blending_mode = mode;

  if (requires_new_canvas) {
    this.sendNewCanvasRequest();
  }
};

// nvidia.Controller.prototype.setServerSideGeometry = function(enabled) {
//   const requires_new_canvas = enabled && !this.server_side_geometry;
//   if (requires_new_canvas) {
//     this.sendNewCanvasRequest();
//   }
//   // TODO: should wait in setting this until server confirms
//   this.server_side_geometry = enabled;
// };

nvidia.Controller.prototype.setDemoMode = function(enabled, demo_seeds) {
  this.demo_mode = enabled;
  this.demo_seeds = demo_seeds;

  if (this.demo_mode) {
    // $("#color1-cont").hide();
    $("#color2-cont").hide();
    $("#auto-new-layer").click();
    $("#hide-stroke").click();
    $('#render-mode').val('clear').trigger('change');
    $('#feature-blending').val('2').trigger('change');
    $('#uvs-mapping').click();
  }
};

nvidia.Controller.prototype.enableLayers = function(baked_bg_canvas_sel, baked_fg_canvas_sel) {
  this.baked_bg_canvas_sel = baked_bg_canvas_sel;
  this.baked_fg_canvas_sel = baked_fg_canvas_sel;
};


nvidia.Controller.prototype.bakeLayers = function(fg_url, bg_url) {
  if (this.baked_bg_canvas_sel === null) {
    nvidia.util.timed_log('Layers not enabled; cannot bake layer.');
  }

  let me = this;

  // Composite foreground canvas onto baked foreground canvas
  let promise0 = new Promise(function(resolve) {
    let fg_img = new Image();
    fg_img.onload = function() {
      let canvas = $(me.baked_fg_canvas_sel)[0];
      canvas.getContext("2d").drawImage(fg_img, 0, 0, canvas.width, canvas.height);
      resolve($(me.baked_fg_canvas_sel)[0].toDataURL('image/png'));
    };
    fg_img.src = fg_url;
  });

  let promise1 = new Promise(function(resolve) {
    let bg_img = new Image();
    bg_img.onload = function () {
      let canvas = $(me.baked_bg_canvas_sel)[0];
      canvas.getContext("2d").drawImage(bg_img, 0, 0, canvas.width, canvas.height);
      resolve($(me.baked_bg_canvas_sel)[0].toDataURL('image/png'));
    };
    bg_img.src = bg_url;
  });
  return Promise.all([promise0, promise1]);
};

nvidia.Controller.prototype.newLayer = function() {
  let fg_url = $(this.fg_canvas_sel)[0].toDataURL('image/png');
  let bg_url = $(this.bg_canvas_sel)[0].toDataURL('image/png');

  let me = this;
  this.bakeLayers(fg_url, bg_url).then(
    value => {me.addToUndoStack(value[0], value[1]); me.clearLayer();});
};

nvidia.Controller.prototype.addToUndoStack = function(fg_url, bg_url) {
  nvidia.util.timed_log('Adding to undo stack: ' + this.undoStack.length);
  while (this.undoStack.length >= this.maxUndos) {
    this.undoStack.shift();
  }
  this.undoStack.push([fg_url, bg_url]);
};

nvidia.Controller.prototype.undo = function() {
  if (this.undoStack.length < 2) {
    nvidia.util.timed_log('No undos left');
    return;
  }
  // Note: top of the undo stack contains the "now",
  // next items contains the past
  let fg_bg_urls = this.undoStack.pop();
  this.redoStack.push(fg_bg_urls);

  fg_bg_urls = this.undoStack[this.undoStack.length - 1];
  this.clearCanvas();
  this.bakeLayers(fg_bg_urls[0], fg_bg_urls[1]);
};

nvidia.Controller.prototype.redo = function() {
  if (this.redoStack.length == 0) {
    nvidia.util.timed_log('No undos left');
    return;
  }
  let fg_bg_urls = this.redoStack.pop();
  this.undoStack.push(fg_bg_urls);

  this.clearCanvas();
  this.bakeLayers(fg_bg_urls[0], fg_bg_urls[1]);

};

nvidia.Controller.prototype.clearCanvas = function() {
  this.clearLayer();

  let canvas = $(this.baked_bg_canvas_sel)[0];
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

  canvas = $(this.baked_fg_canvas_sel)[0];
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
};

nvidia.Controller.prototype.clearLayer = function() {
  this.ctx.clearRect(0, 0, this.render_canvas[0].width, this.render_canvas[0].height);
  this.drawController.clearCanvas();

  if (this.server_side_geometry || this.feature_blending_mode > 0) {
    this.sendNewCanvasRequest();
  }
};

/**
 * Initializes controller; should only run once.
 */
nvidia.Controller.prototype.init = function() {
  this.initWebSocket();
  this.initEvents();
  this.newLayer();
};

nvidia.Controller.prototype.getSeed = function() {
  const seed_str = $("#seed").val();
  if (seed_str.length === 0) {
    return undefined;
  }
  return Number(seed_str);
};

nvidia.Controller.prototype.requestNewBrush = function() {
  let seed = this.getSeed();
  if (seed === this.current_seed) {
    seed = undefined;
  }
  if (seed === undefined && this.demo_mode && this.demo_seeds && this.demo_seeds.length > 0) {
    seed = this.demo_seeds[(this.demo_seed_id % this.demo_seeds.length)];
    this.demo_seed_id += 1;
  }
  this.requestBrush(seed);
};

nvidia.Controller.prototype.requestSaveBrush = function() {
  let request = {
    type: "save_brush"
  };
  console.log("Sending save brush request.");
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.sendNewCanvasRequest = function() {
  if (!this.ws_ready) {
    console.log("Socket is not ready; cannot send new canvas request");
    return;
  }
  let request = {
    type: "new_canvas",
    rows: $(this.fg_canvas_sel).attr("height"),
    cols: $(this.fg_canvas_sel).attr("width"),
    feature_blending: this.feature_blending_mode
  };
  console.log("Sending new canvas request.");
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.setUvsAdjustment = function(enabled) {
  let request = {
    type: "set_option",
    option: "uvs_mapping",
    value: enabled
  };
  console.log("Sending option request.");
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.setUsePositions = function(enabled) {
  let request = {
    type: "set_option",
    option: "positions",
    value: enabled
  };
  console.log("Sending option request.");
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.requestBrush = function(seed) {
  let request = {
    type: "set_brush"
  };
  if (seed !== undefined) {
    request.seed = seed;
  }
  console.log("Sending brush request with seed: " + seed);
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.requestBrushById = function(library_id, style_id) {
  let request = {
    type: "set_brush",
    library_id: library_id,
    style_id: style_id
  };
  console.log("Sending brush request with style_id: " + style_id +
              ", library " + library_id);
  this.ws.send(JSON.stringify(request));
};

nvidia.Controller.prototype.setRenderMode = function(mode) {
  if (!this.ws_ready) {
    console.log("Socket is not ready; cannot set render mode");
    return;
  }
  let request = {
    type: "set_render_mode"
  };
  request.mode = mode;
  console.log("Sending render mode request with mode: " + mode);
  this.ws.send(JSON.stringify(request));
};

/**
 * Initializes websocket connection and message callbacks.
 */
nvidia.Controller.prototype.initWebSocket = function() {
  this.ws = new WebSocket("ws://" + window.location.host + "/websocket/"); //window.location.pathname);
  this.ws.binaryType = "arraybuffer";

  this.ws.onopen = function(c){return function() { c.onopen();} }(this);
  this.ws.onmessage = function(c){return function(evt) { c.onmessage(evt);} }(this);
  this.ws.onclose = function(c){return function() { c.onclose();} }(this);
};

nvidia.Controller.prototype.toggleBackground = function() {
  $(this.bg_canvas_sel).toggleClass("transparent");
  if (this.baked_bg_canvas_sel) {
    $(this.baked_bg_canvas_sel).toggleClass("transparent");
  }
};

nvidia.Controller.prototype.toggleForeground = function() {
  $(this.fg_canvas_sel).toggleClass("transparent");
  if (this.baked_fg_canvas_sel) {
    $(this.baked_fg_canvas_sel).toggleClass("transparent");
  }
};

/**
 * Initializes all the UI events.
 */
nvidia.Controller.prototype.initEvents = function() {
  $(window).on('touchmove', function(e) {
      e.preventDefault();
  });

  let me = this;
  this.drawController.canvas.on(
    "window_complete", (e, d) => {me.onDrawingWindowComplete(e, d)});
  this.drawController.canvas.on(
    "force", (e, d) => {me.onForceEventDemo(e, d)});

  $("#clear-layer").click(function(e) { me.clearLayer(); });
  $("#clear-canvas").click(function(e) { me.clearCanvas(); });
  $("#new-layer").click(function(e) { me.newLayer(); });
  $("#new-brush").click(function(e) {
    me.clearLayer();
    me.requestNewBrush();
  });
  $("#save-brush").click(function(e) { me.requestSaveBrush(); });
  $("#brush").click(function(e) {
    console.log("Brush click");
    $("#brush-menu").toggle();
  });
  $(".brushinfo").click(function(e) {
    let lib_brush = this.id.split(":");
    me.requestBrushById(lib_brush[0], lib_brush[1]);
  });
  $("#undo").click(function(e) { me.undo(); });
  $("#redo").click(function(e) { me.redo(); });
  $("#render-mode").change(function(e) {
    var selectedVal = $(this).val();
    me.setRenderMode(selectedVal);
  });

  // Debug panel
  $("#show-debug").click(function(e) { $("#debug").toggle(); });
  $("#hide-debug").click(function(e) { $("#debug").hide(); });
  $("#debug-show-check").on("change", function(e) {
    $("#debug-img-container").toggleClass("checkered");
  });
  $("#hide-stroke").on("change", function(e) {
    me.toggleBackground();
  });
  $("#hide-canvas").on("change", function(e) {
    me.toggleForeground();
  });
  $("#feature-blending").change(function(e) {
    var selectedVal = $(this).val();
    me.setFeatureBlending(selectedVal);
  });
  $("#crop-margin").change(function(e) {
    var selectedVal = $(this).val();
    me.crop_margin = Number(selectedVal);
  });
  this.crop_margin = Number($("#crop-margin").val());

  // $("#use-server-geo").on("change", function(e) {
  //   me.setServerSideGeometry(this.checked);
  // });
  $("#use-positions").on("change", function(e) {
    me.setUsePositions(this.checked);
  });

  $("#uvs-mapping").on("change", function(e) {
    me.setUvsAdjustment(this.checked);
  });

  $("#auto-new-layer").on("change", function(e) {
    me.auto_new_layer = this.checked;
  });

  // Brush control
  $("#thickness").val(this.drawController.thickness);
  $("#thickness").on("input", function(e) {
    $("#thickness-val").text($(this).val());
    me.drawController.thickness = Number($(this).val());
  });

  $("#download-all").click(function(e) {
    me.downloadAll($("#prefix").val());
  });
  $("#download-stroke").click(function(e) {
    me.downloadStroke($("#prefix").val());
  });

  $("#color0").on("change", function(e) {
    if (me.auto_new_layer) {
      nvidia.util.timed_log("New layer");
      me.newLayer();
    }
  });

  $("#color1").on("change", function(e) {
    if (me.auto_new_layer) {
      nvidia.util.timed_log("New layer");
      me.newLayer();
    }
  });
};

/**
 * Sets patch width of the patch sent to the server.
 */
nvidia.Controller.prototype.setPatchWidth = function(patch_width) {
  nvidia.util.timed_log("Setting patch width: " + patch_width, "DEBUG");
  this.patch_width = patch_width;
  this.drawController.patch_width = Math.floor(patch_width / 2);
};

/**
 * Handles event from the drawing controller that runs when the user
 * stroke exits the drawing patch/window of a certain size.
 *
 * @param {Event} e event object
 * @param {Object} data custom data associated with the event
 */
nvidia.Controller.prototype.onDrawingWindowComplete = function(e, data) {
  if (data['dirty'].maxDim() > 0) {
    let req = this.encodeDrawingRequest(data);
    this.ws.send(req);  // Send over web socket
  } else if (data.hasOwnProperty('stroke_end')) {
    nvidia.util.timed_log('Setting timeout for stroke end.');
    setTimeout(function(me) { return function() { me.onDrawingStrokeComplete(); } }(this),
               300);
  }
};

nvidia.Controller.prototype.enableForceDemo = function(brush_lib, style1, style2) {
  if (brush_lib) {
    this.auto_new_layer = false;
    this.drawController.trigger_force_event = true;
    this.forcedemo = {
      brush_lib: brush_lib,
      style1: style1,
      style2: style2
    };
  } else {
    this.forcedemo = null;
  }
};

nvidia.Controller.prototype.onForceEventDemo = function(e, data) {
  if (this.forcedemo === null) {
    return;
  }

  let force = data.force;

  let incr = Math.floor(force * 1.8 * 10);
  console.log('Incr ' + incr + " force " + force);
  let brush_id = null;
  if (incr <= 0) {
    brush_id = this.forcedemo.style1;
  } else if (incr > 9) {
    brush_id = this.forcedemo.style2;
  } else {
    brush_id = this.forcedemo.style1 + "_0_" + (10 - incr) + "_" + this.forcedemo.style2;
  }

  if (this.forcedemo.brush_lib + "_" + brush_id != this.current_brush_id) {
    console.log("Requesting force-based " + brush_id + " for force " + force);
    this.requestBrushById(this.forcedemo.brush_lib, brush_id);
  }
};

nvidia.Controller.prototype.onDrawingStrokeComplete = function(e, data) {
  nvidia.util.timed_log('Stroke complete');
  if (this.auto_new_layer) {
    //this.newLayer();
  }
};

/**
 * Encodes drawing request metadata.
 * Layout:
 * uint8:
 *    [0] - 0/1 is debug
 *    [1] - 0/1/2/3 number of custom colors
 *    [2] - int encoding any extra data
 *    3 * number colors uints for primary/secondary colors if included
 *
 * @return {Uint8Array}
 */
nvidia.Controller.prototype.encodeDrawingMetadata = function(extra_data) {
  // if (this.demo_mode) {
  //   return this.encodeDrawingMetadataDemo();
  // }

  let ncolors = 0;
  for (let i = 0; i < 3; ++i) {
    if ($("#color" + i + "-check")[0].checked === true) {
      ncolors += 1;
    }
  }
  let metadata = new Uint8Array(3 + ncolors * 4);
  metadata[0] = ($("#debug-check")[0].checked === true) ? 1 : 0;
  metadata[1] = ncolors;
  metadata[2] = extra_data;

  let midx = 3;
  for (let i = 0; i < 3; ++i) {
    if ($("#color" + i + "-check")[0].checked === true) {
      metadata[midx] = i;

      const rgb = tinycolor($("#color" + i).val()).toRgb();
      metadata[midx + 1] = rgb.r;
      metadata[midx + 2] = rgb.g;
      metadata[midx + 3] = rgb.b;
      midx += 4;
    }
  }

  return metadata;
};


// Use primary color as both first colors
nvidia.Controller.prototype.encodeDrawingMetadataDemo = function() {
  let ncolors = 0;
  if ($("#color0-check")[0].checked === true) {
    ncolors = 2;
  }
  let metadata = new Uint8Array(2 + ncolors * 4);
  metadata[0] = ($("#debug-check")[0].checked === true) ? 1 : 0;
  metadata[1] = ncolors;

  let midx = 2;
  if ($("#color0-check")[0].checked === true) {
    let color0 = tinycolor($("#color0").val());
    for (let i = 0; i < 2; ++i) {
      let rgb = color0.toRgb();
      if (i == 1) {
        let adjusted_hsv = color0.toHsv();
        adjusted_hsv.v = adjusted_hsv.v * 0.8;
        rgb = tinycolor(adjusted_hsv).toRgb();
      }
      metadata[midx] = i;
      metadata[midx + 1] = rgb.r;
      metadata[midx + 2] = rgb.g;
      metadata[midx + 3] = rgb.b;
      midx += 4;
    }
  }

  return metadata;
};


/**
 * Encodes a request to the server for a rendered stroke. Includes
 * patches of both background and foreground canvasses in the
 * request. Current binary format is:
 * bytes ?: (see encodeDrawingMetadata)
 * bytes 0-4: patch_width
 * bytes 4-8: patch_height
 * bytes 8-12: x-center of the patch in the canvas coordinate frame
 * bytes 12-14: y-center of the patch in the canvas coordinate frame
 * next block: background canvas bytes as uint8 (4 channels)
 * next block: final drawing canvas bytes as uint8 (4 channels)
 *
 * @param {Number} x x-center of the patch in the canvas coordinate frame
 * @param {Number} y y-center of the patch in the canvas coordinate frame
 *
 * @return {ArrayBuffer} ready to be sent over web socket
 */
nvidia.Controller.prototype.encodeDrawingRequest = function(data) {
  const bbox = data.dirty;
  const center = bbox.getCenter();
  let half_width = Math.floor(this.patch_width / 2);
  let x = Math.min(Math.max(0, center[0] - half_width), $(this.fg_canvas_sel).attr("width") - this.patch_width - 1);
  let y = Math.min(Math.max(0, center[1] - half_width), $(this.fg_canvas_sel).attr("height") - this.patch_width - 1);
  let patch_width = this.patch_width;
  let patch_height = this.patch_width;

  // send smaller patch if geometry is on the server
  if (this.server_side_geometry) {
    x = bbox.min_x;
    y = bbox.min_y;
    patch_width = bbox.getWidth();
    patch_height = bbox.getHeight();
  }

  // Overall metadata
  let extra_data = 0;
  if (data.hasOwnProperty('stroke_end')) {
    extra_data = 10;
  }
  let drawing_metadata = this.encodeDrawingMetadata(extra_data);

  // Note: out of bounds allowed
  let stroke_data = this.drawController.ctx.getImageData(
    x, y, patch_width, patch_height);

  // Note: in current implementation we do not need to send rendered canvas
  // let fg_data = this.ctx.getImageData(
  //  x, y, patch_width, patch_height);

  // To the extra line above we add metadata
  let metadata = new Int32Array(5);
  metadata[0] = stroke_data.width;
  metadata[1] = stroke_data.height;
  metadata[2] = x;
  metadata[3] = y;
  metadata[4] = this.crop_margin;

  return nvidia.util.catArrayBuffers(
    [drawing_metadata.buffer,
     metadata.buffer,
     stroke_data.data.buffer]);
};

/**
 * Decodes response from the server that renders the requested stroke
 * patch.
 */
nvidia.Controller.prototype.decodeDrawingResponse = function(binary_data, offset) {
  nvidia.util.timed_log("Parsing binary data");
  const meta = new Int32Array(binary_data, offset, 4);

  let res = {
    width: meta[0],
    height: meta[1],
    x: meta[2],
    y: meta[3]
  };
  res.image = new Uint8ClampedArray(
    binary_data, offset + 4 * 4, res.width * res.height * 4);
  return res;
};

/**
 * Handles the event fired when web socket connection is successfully
 * initialized.
 */
nvidia.Controller.prototype.onopen = function() {
  nvidia.util.timed_log("Connection open");
  this.ws_ready = true;

  this.clearCanvas();
  if (this.demo_mode) {
    this.setDemoMode(true, []);
    this.requestNewBrush();
  }
};

/**
 * Handles the event fired when web socket connection is closed
 * (for example, when the server encounters an error).
 */
nvidia.Controller.prototype.onclose = function() {
  nvidia.util.timed_log('Web socket connection closed', 'WARN');
  this.ws_ready = false;
};

/**
 * Handles incoming web socket messages.
 */
nvidia.Controller.prototype.onmessage = function (evt) {
  // TODO: ensure this part happens in the background, if not already so
  // nvidia.util.timed_log("Message received");

  if (typeof evt.data === "string") {
    nvidia.util.timed_log("Got text message: " + evt.data);
    this.handleTextMessage(evt.data);
  } else if (evt.data instanceof ArrayBuffer) {
    nvidia.util.timed_log("Got arraybuffer message");
    this.handleBinaryMessage(evt.data);
  } else {
    nvidia.util.timed_log("Got unknown message");
    console.log(evt);
  }
};

nvidia.Controller.prototype.onNewBrushSet = function(data) {
  if (data.hasOwnProperty('colors')) {
    let colors = data.colors.split(':');
    for (let i = 0; i < colors.length; ++i) {
      $("#color" + i).val('#' + tinycolor(colors[i]).toHex());
    }
  }
  $(".brushinfo").removeClass("selected");
  if (data.library_id) {
    $(document.getElementById(data.library_id + ":" + data.style_id)).addClass("selected");
  }

  try {
    this.current_seed = Number(data.style_id);
    $("#seed").val(data.style_id);
  } catch(error) {
    this.current_seed = -1;
    $("#seed").val('');
  }

  if (this.auto_new_layer) {
    nvidia.util.timed_log("New layer");
    this.newLayer();
  }
  this.current_brush_id = data.library_id + '_' + data.style_id;
};

/**
 * Handles incoming text message.
 */
nvidia.Controller.prototype.handleTextMessage = function(message_data) {
  const message = JSON.parse(message_data);
  nvidia.util.timed_log("Got message:" + message);

  if (message["type"] === "modelinfo" && message["data"]) {
    this.setPatchWidth(Number(message["data"]["patch_width"]));
  } else if (message["type"] === "brushinfo" && message["data"]) {
    this.onNewBrushSet(message["data"]);
  } else {
    nvidia.util.timed_log("Unexpected message:");
  }
};

/**
 * Handles incoming binary message.
 */
nvidia.Controller.prototype.handleBinaryMessage = function(binary_data) {
  const response_type = new Int32Array(binary_data, 0, 1)[0];

  if (response_type === 0 || response_type === 10) {
    let data = this.decodeDrawingResponse(binary_data, 4);
    // nvidia.util.timed_log("Rendering patch " + data.height + "x" + data.width +
    //                      " at (x = " + data.x + ", y = " + data.y + ")");
    let img_data = new ImageData(data.image, data.width, data.height);
    this.ctx.putImageData(img_data, data.x, data.y,
                          0, 0, data.width, data.height);
    if (response_type === 10) {
      this.onDrawingStrokeComplete();
    }
  } else if (response_type === 1) {
    // Debug data
    let data = this.decodeDrawingResponse(binary_data, 4);
    let img_data = new ImageData(data.image, data.width, data.height);
    let url = nvidia.util.imageDataToURL(img_data);
    $("#debug-img")[0].src = url;
    // nvidia.util.downloadURL('debug.png', url);
  } else if (response_type === 2) {
    // Brush info
  }
};
// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
var nvidia = nvidia || {};

nvidia.TouchManager = function(pageToPos) {
  this.pageToPos = pageToPos || function(pageX, pageY) { return [pageX, pageY]; };
  this.stylusOnly = false;

  this.touches = {};
  this.active_touch = null;
};

nvidia.TouchManager.prototype.numRegisteredTouches = function() {
  return nvidia.util.getOwnKeys(this.touches).length;
};

nvidia.TouchManager.prototype.extractChangedTouches = function(e) {
  var res = []
  nvidia.util.ensurePageX(e);
  if (!e.changedTouches) {
    res.push(["mouse", e.pageX, e.pageY, null]);
  } else {
    for (var i = 0; i < e.changedTouches.length; ++i) {
      var touch = e.changedTouches[i];
      res.push([touch.identifier, touch.pageX, touch.pageY, touch]);
    }
  }
  return res;
};

nvidia.TouchManager.prototype.registerTouches = function(e) {
  var touches = this.extractChangedTouches(e);
  for (var i = 0; i < touches.length; ++i) {
    var touchInfo = touches[i];
    this.touches[touchInfo[0]] = {
      startPos: this.pageToPos(touchInfo[1], touchInfo[2]),
      lastPos: this.pageToPos(touchInfo[1], touchInfo[2]),
      force: touchInfo[3].force
    };
    if (this.active_touch === null &&
        (!this.stylusOnly || touchInfo[3].force > 0)) {
      this.active_touch = touchInfo[0];
    }
  }
};

nvidia.TouchManager.prototype.updateTouches = function(e) {
  var touches = this.extractChangedTouches(e);
  let active_touch_updated = false;
  for (var i = 0; i < touches.length; ++i) {
    var touchInfo = touches[i];

    if (touchInfo[0] in this.touches) {
      this.touches[touchInfo[0]].lastPos = this.pageToPos(touchInfo[1], touchInfo[2]);
      this.touches[touchInfo[0]].force = touchInfo[3].force;
      if (touchInfo[0] === this.active_touch) {
        active_touch_updated = true;
      }
    }
  }
  return active_touch_updated;
};

nvidia.TouchManager.prototype.unregisterTouch = function(touchid) {
  if (touchid in this.touches) {
    //console.log("Unregistering touch " + touchid);
    delete this.touches[touchid];
  }
  if (touchid === this.active_touch) {
    this.active_touch = null;
  }
};

nvidia.TouchManager.prototype.unregisterTouches = function(e) {
  var touches = this.extractChangedTouches(e);
  for (var i = 0; i < touches.length; ++i) {
    var touchInfo = touches[i];

    this.unregisterTouch(touchInfo[0]);
  }

  // If we missed some touches (unclear why this happens)
  var ids = new Set(nvidia.TouchManager.getTouchIds(e, e.touches));
  var keys = nvidia.util.getOwnKeys(this.touches);
  for (var i = 0; i < keys.length; ++i) {
    if (!ids.has(keys[i])) {
      this.unregisterTouch(keys[i]);
    }
  }
};

nvidia.TouchManager.prototype.getActiveTouchPosition = function() {
  if (this.active_touch === null) {
    return null;
  }

  let info = this.getRegisteredTouchInfo(this.active_touch);
  return {pageX: info.lastPos[0], pageY: info.lastPos[1]}
};

nvidia.TouchManager.prototype.getActiveTouchInfo = function() {
  if (this.active_touch === null) {
    return null;
  }

  return this.getRegisteredTouchInfo(this.active_touch);
};


nvidia.TouchManager.prototype.getRegisteredTouchInfo = function(id) {
  if (id in this.touches) {
    return this.touches[id];
  }
  nvidia.util.timed_log("Cannot find touch id " + id);
  console.log(this.touches);
  return null;
};

nvidia.TouchManager.prototype.isTouchRegistered = function(id) {
  return id in this.touches;
};

nvidia.TouchManager.numTouches = function(e) {
  if (e.touches) {
    return e.touches.length;
  } else {
    return 1;
  }
};

nvidia.TouchManager.getSoleTouchId = function(e, touchlist) {
  var ids = nvidia.TouchManager.getTouchIds(e, touchlist);
  if (ids.length == 1) {
    return ids[0];
  }
  return null;
};

nvidia.TouchManager.getTouchIds = function(e, touchlist) {
  var res = [];
  if (!e.touches) {
    res.push("mouse");
  } else if (touchlist) {
    for (var i = 0; i < touchlist.length; ++i) {
      res.push(touchlist[i].identifier);
    }
  }
  return res;
};
// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

var nvidia = nvidia || {};
nvidia.util = nvidia.util || {};

nvidia.util.LOG_LEVELS = { DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40 };
nvidia.util.GLOBAL_LOG_LEVEL = nvidia.util.LOG_LEVELS.DEBUG;

nvidia.util.set_global_log_level = function(level_name) {
    nvidia.util.GLOBAL_LOG_LEVEL = nvidia.util.LOG_LEVELS[level_name];
};

nvidia.util.toggleLib = function(lib_id) {
    $("#" + lib_id).toggle();
    $("#" + lib_id + "-arrow").toggleClass('up').toggleClass('down');
};

/**
 * Log message to the console and append current time.
 *
 * @param {String} level_name one of DEEBUG, INFO, WARN, ERROR (or None)
 */
nvidia.util.timed_log = function(message, level_name) {
  let level = nvidia.util.LOG_LEVELS['DEBUG'];
  if (level_name && nvidia.util.LOG_LEVELS[level_name]) {
    level = nvidia.util.LOG_LEVELS[level_name];
   }
   if (level >= nvidia.util.GLOBAL_LOG_LEVEL) {
     let d = new Date();
     let msg = d.getMinutes() + ":" + d.getSeconds() + ":" + d.getMilliseconds() + "  " + message;
     if (level >= nvidia.util.LOG_LEVELS['ERROR']) {
        console.error(msg);
     } else if (level >= nvidia.util.LOG_LEVELS['WARN']) {
        console.warn(msg);
     } else if (level >= nvidia.util.LOG_LEVELS['INFO']) {
        console.info(msg);
     } else {
        console.log(msg);
     }
   }
};

/**
 * Detects native byte order (may be needed to diagnose binary
 * decoding issues).
 */
nvidia.util.detect_native_byteorder = function() {
    let array_uint32 = new Uint32Array([0x11223344]);
    let array_uint8 = new Uint8Array(array_uint32.buffer);

    if (array_uint8[0] === 0x44) {
        return 'little';
    } else if (array_uint8[0] === 0x11) {
        return 'big';
    } else {
        return 'unknown';
    }
};

/**
 * Issues a download request for a data URL; useful for debugging and
 * more. E.g. downloadURL('image.png', $('bg-canvas')[0].toDataURL()).
 *
 * @param {String} filename the name of the file to save download as
 * @param {String} url data URL encoding the data
 */
nvidia.util.downloadURL = function(filename, url) {
  var a = document.createElement("a");
  document.body.appendChild(a);
  a.style = "display: none";
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
};

/**
 * Concatenates input array buffers.
 */
nvidia.util.catArrayBuffers = function(arrays) {
  let lengths = 0;
  for (let i = 0; i < arrays.length; ++i) {
    lengths += arrays[i].byteLength;
  }
  var tmp = new Uint8Array(lengths);
  let start = 0;
  for (let i = 0; i < arrays.length; ++i) {
    tmp.set(new Uint8Array(arrays[i]), start);
    start += arrays[i].byteLength;
  }
  return tmp.buffer;
};


// From: https://developer.mozilla.org/en-US/docs/Web/API/Blob
nvidia.util.typedArrayToURL = function(typedArray, mimeType) {
  return URL.createObjectURL(new Blob([typedArray.buffer], {type: mimeType}))
};

nvidia.util.imageDataToURL = function(imgData) {
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL();
};

nvidia.util.ensurePageX = function(event) {
  //console.log(event);
  if (event.pageX === undefined) {
    if (event.originalEvent && event.originalEvent.pageX !== undefined) {
      event.pageX = event.originalEvent.pageX;
      event.pageY = event.originalEvent.pageY;
    }
  }
  //console.log('ok');
};

nvidia.util.getOwnKeys = function(dict) {
  var res = [];
  for (var key in dict) {
    if (dict.hasOwnProperty(key)) {
      res.push(key);
    }
  }
  return res;
};

nvidia.util.getOwnValues = function(dict) {
  var keys = nvidia.util.getOwnKeys(dict);
  var res = new Array(keys.length);
  for (var i = 0; i < keys.length; ++i) {
    res[i] = dict[keys[i]];
  }
  return res;
};


if (typeof module !== 'undefined') {
    module.exports = nvidia.util;
}
