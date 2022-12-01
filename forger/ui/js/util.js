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
