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
