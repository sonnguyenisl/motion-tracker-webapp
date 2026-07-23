// Result page: toggle the recording between the raw and skeleton-burned video.

(function () {
  const btn = document.getElementById("btnToggleSkeleton");
  const video = document.getElementById("resultVideo");
  if (!btn || !video) return;

  let showingSkeleton = false;

  btn.addEventListener("click", () => {
    const raw = video.dataset.raw;
    const skeleton = video.dataset.skeleton;
    if (!skeleton) return;

    const t = video.currentTime;
    const wasPlaying = !video.paused;

    showingSkeleton = !showingSkeleton;
    video.src = showingSkeleton ? skeleton : raw;
    btn.textContent = "Skeleton: " + (showingSkeleton ? "On" : "Off");

    video.addEventListener("loadedmetadata", function once() {
      video.currentTime = t;
      if (wasPlaying) video.play().catch(() => {});
      video.removeEventListener("loadedmetadata", once);
    });
  });
})();
