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
