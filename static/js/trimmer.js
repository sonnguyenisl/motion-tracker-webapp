/* Aximove Video Trimmer
 *
 * A reusable, dark-themed video trimmer with a timeline track, two draggable
 * handles, and a highlighted selection region. Works with both mouse and touch.
 *
 * Usage:
 *   var trimmer = new VideoTrimmer(videoElement, {
 *     onTrimChange: function (start, end) { ... },
 *   });
 *   trimmer.destroy();  // clean up when done
 */

(function (global) {
  "use strict";

  var TEMPLATE =
    '<div class="ft-trimmer bg-gray-900/90 border border-gray-700 rounded-2xl p-4 shadow-xl">' +
      '<div class="flex items-center justify-between mb-3">' +
        '<span class="text-xs font-bold uppercase tracking-wider text-orange-400">Trim Video</span>' +
        '<div class="flex items-center gap-3 text-xs text-gray-400">' +
          '<span><span class="text-gray-300 font-bold" id="ftTrimStartLabel">0:00</span> start</span>' +
          '<span><span class="text-gray-300 font-bold" id="ftTrimEndLabel">0:00</span> end</span>' +
          '<span class="text-gray-500">|</span>' +
          '<span class="text-orange-400 font-bold" id="ftTrimDuration">0:00</span>' +
        '</div>' +
      '</div>' +
      '<div class="ft-timeline-wrap relative h-12 bg-gray-800 rounded-xl border border-gray-700 overflow-hidden cursor-pointer select-none" id="ftTimelineWrap">' +
        '<div class="ft-timeline-bg absolute inset-0 flex" id="ftTimelineBg"></div>' +
        '<div class="ft-selection absolute top-0 bottom-0 bg-orange-500/20 border-l-2 border-r-2 border-orange-500 pointer-events-none" id="ftSelection"></div>' +
        '<div class="ft-handle ft-handle-left absolute top-0 bottom-0 w-4 -ml-2 cursor-ew-resize z-10 flex items-center justify-center" id="ftHandleLeft">' +
          '<div class="w-1 h-full bg-orange-500 rounded-full shadow-lg"></div>' +
        '</div>' +
        '<div class="ft-handle ft-handle-right absolute top-0 bottom-0 w-4 -ml-2 cursor-ew-resize z-10 flex items-center justify-center" id="ftHandleRight">' +
          '<div class="w-1 h-full bg-orange-500 rounded-full shadow-lg"></div>' +
        '</div>' +
      '</div>' +
      '<div class="flex items-center justify-between mt-3">' +
        '<button type="button" class="ft-trim-play-btn flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider bg-gray-800 border border-gray-700 text-gray-300 hover:text-orange-400 hover:border-orange-500/50 transition" id="ftTrimPlayBtn">' +
          '<svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>' +
          'Preview' +
        '</button>' +
        '<div class="flex items-center gap-2">' +
          '<button type="button" class="ft-trim-reset-btn px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider bg-gray-800 border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-500 transition" id="ftTrimResetBtn">Reset</button>' +
        '</div>' +
      '</div>' +
    '</div>';

  function fmt(sec) {
    var m = Math.floor(sec / 60);
    var s = Math.floor(sec % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  var PLAY_BTN_HTML =
    '<svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Preview';
  var PAUSE_BTN_HTML =
    '<svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg> Pause';

  function VideoTrimmer(video, opts) {
    this.video = video;
    this.opts = opts || {};
    // Max selectable window length in seconds (0 = unlimited). When set, the
    // selection can never span more than this; if the clip is longer the
    // initial window is capped to it (defaulting to the first N seconds).
    this.maxDuration = this.opts.maxDuration || 0;
    this.start = 0;
    this.end = video.duration || 0;
    if (this.maxDuration && this.end > this.maxDuration) {
      this.end = this.maxDuration;
    }
    this.dragging = null;  // 'left' | 'right' | null
    this._destroyed = false;

    this._build();
    this._renderTimeline();
    this._bindEvents();
  }

  VideoTrimmer.prototype._build = function () {
    var wrap = document.createElement("div");
    wrap.className = "ft-trimmer-wrap mt-4";
    wrap.innerHTML = TEMPLATE;
    this.wrap = wrap;

    this.timeline = wrap.querySelector("#ftTimelineWrap");
    this.timelineBg = wrap.querySelector("#ftTimelineBg");
    this.selection = wrap.querySelector("#ftSelection");
    this.handleLeft = wrap.querySelector("#ftHandleLeft");
    this.handleRight = wrap.querySelector("#ftHandleRight");
    this.startLabel = wrap.querySelector("#ftTrimStartLabel");
    this.endLabel = wrap.querySelector("#ftTrimEndLabel");
    this.durationLabel = wrap.querySelector("#ftTrimDuration");
    this.playBtn = wrap.querySelector("#ftTrimPlayBtn");
    this.resetBtn = wrap.querySelector("#ftTrimResetBtn");

    // Insert after the video element.
    this.video.parentNode.insertBefore(wrap, this.video.nextSibling);
  };

  VideoTrimmer.prototype._renderTimeline = function () {
    var dur = this.video.duration || 1;
    var leftPct = (this.start / dur) * 100;
    var rightPct = (this.end / dur) * 100;

    this.selection.style.left = leftPct + "%";
    this.selection.style.width = (rightPct - leftPct) + "%";
    this.handleLeft.style.left = leftPct + "%";
    this.handleRight.style.left = rightPct + "%";

    this.startLabel.textContent = fmt(this.start);
    this.endLabel.textContent = fmt(this.end);
    this.durationLabel.textContent = fmt(this.end - this.start);

    if (this.opts.onTrimChange) {
      this.opts.onTrimChange(this.start, this.end);
    }
  };

  VideoTrimmer.prototype._getPos = function (e) {
    var rect = this.timeline.getBoundingClientRect();
    var clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return (clientX - rect.left) / rect.width;
  };

  // Seek the video to a given time so the frame under the handle is visible
  // while scrubbing. Pauses playback (we're scrubbing, not playing) and skips
  // redundant seeks to keep dragging smooth.
  VideoTrimmer.prototype._seek = function (sec) {
    if (!isFinite(sec)) return;
    var dur = this.video.duration || 0;
    sec = Math.max(0, dur ? Math.min(sec, dur) : sec);
    if (!this.video.paused) {
      this.video.pause();
      this.playBtn.innerHTML = PLAY_BTN_HTML;
    }
    if (Math.abs(this.video.currentTime - sec) > 0.01) {
      this.video.currentTime = sec;
    }
  };

  // Keep the selected window within maxDuration. The handle the user is moving
  // stays put; the opposite handle slides in to honour the cap, so the window
  // can be dragged anywhere along the clip without ever exceeding the limit.
  VideoTrimmer.prototype._enforceMax = function (moved) {
    if (!this.maxDuration || (this.end - this.start) <= this.maxDuration) return;
    var dur = this.video.duration || 0;
    if (moved === "left") {
      this.end = dur ? Math.min(dur, this.start + this.maxDuration)
                     : this.start + this.maxDuration;
    } else {
      this.start = Math.max(0, this.end - this.maxDuration);
    }
  };

  VideoTrimmer.prototype._onPointerDown = function (e) {
    if (this._destroyed) return;
    var target = e.target.closest(".ft-handle");
    if (target) {
      this.dragging = target.classList.contains("ft-handle-left") ? "left" : "right";
      // Jump to the grabbed handle's frame straight away.
      this._seek(this.dragging === "left" ? this.start : this.end);
      e.preventDefault();
      return;
    }
    // Click on timeline — move nearest handle.
    var pos = this._getPos(e);
    var dur = this.video.duration || 1;
    var clickSec = pos * dur;
    var distLeft = Math.abs(clickSec - this.start);
    var distRight = Math.abs(clickSec - this.end);
    if (distLeft < distRight) {
      this.start = Math.max(0, Math.min(clickSec, this.end - 0.5));
      this._enforceMax("left");
      this._seek(this.start);
    } else {
      this.end = Math.min(dur, Math.max(clickSec, this.start + 0.5));
      this._enforceMax("right");
      this._seek(this.end);
    }
    this._renderTimeline();
  };

  VideoTrimmer.prototype._onPointerMove = function (e) {
    if (!this.dragging || this._destroyed) return;
    e.preventDefault();
    var pos = this._getPos(e);
    var dur = this.video.duration || 1;
    var sec = pos * dur;

    if (this.dragging === "left") {
      this.start = Math.max(0, Math.min(sec, this.end - 0.5));
      this._enforceMax("left");
      this._seek(this.start);
    } else {
      this.end = Math.min(dur, Math.max(sec, this.start + 0.5));
      this._enforceMax("right");
      this._seek(this.end);
    }
    this._renderTimeline();
  };

  VideoTrimmer.prototype._onPointerUp = function () {
    this.dragging = null;
  };

  VideoTrimmer.prototype._bindEvents = function () {
    var self = this;

    // Mouse
    this.timeline.addEventListener("mousedown", function (e) { self._onPointerDown(e); });
    document.addEventListener("mousemove", function (e) { self._onPointerMove(e); });
    document.addEventListener("mouseup", function () { self._onPointerUp(); });

    // Touch
    this.timeline.addEventListener("touchstart", function (e) { self._onPointerDown(e); }, { passive: false });
    document.addEventListener("touchmove", function (e) { self._onPointerMove(e); }, { passive: false });
    document.addEventListener("touchend", function () { self._onPointerUp(); });

    // Play/Pause preview
    this.playBtn.addEventListener("click", function () {
      if (self.video.paused) {
        self.video.currentTime = self.start;
        self.video.play();
        self.playBtn.innerHTML = PAUSE_BTN_HTML;
      } else {
        self.video.pause();
        self.playBtn.innerHTML = PLAY_BTN_HTML;
      }
    });

    // Stop the preview at the trim end so it only plays the selected region.
    this.video.addEventListener("timeupdate", function () {
      if (self._destroyed) return;
      if (!self.video.paused && self.video.currentTime >= self.end) {
        self.video.pause();
        self.video.currentTime = self.end;
        self.playBtn.innerHTML = PLAY_BTN_HTML;
      }
    });

    // When video reaches end, reset play button
    this.video.addEventListener("ended", function () {
      self.playBtn.innerHTML = PLAY_BTN_HTML;
    });

    // Reset
    this.resetBtn.addEventListener("click", function () {
      self.start = 0;
      self.end = self.video.duration || 0;
      if (self.maxDuration && self.end > self.maxDuration) {
        self.end = self.maxDuration;
      }
      self._renderTimeline();
      self.video.pause();
      self.video.currentTime = 0;
      self.playBtn.innerHTML = PLAY_BTN_HTML;
    });
  };

  VideoTrimmer.prototype.getTrimValues = function () {
    return { start: this.start, end: this.end };
  };

  VideoTrimmer.prototype.destroy = function () {
    this._destroyed = true;
    if (this.wrap && this.wrap.parentNode) {
      this.wrap.parentNode.removeChild(this.wrap);
    }
  };

  global.VideoTrimmer = VideoTrimmer;
})(window);