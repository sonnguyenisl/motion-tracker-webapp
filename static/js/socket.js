// Flask-SocketIO client. Establishes the connection, joins a per-client room,
// and re-broadcasts server progress as window CustomEvents so any page script
// (camera.js) can react without importing the socket directly.
//
// Requires the socket.io client to be loaded globally first (CDN <script>).

let room = null;
const socket = (typeof io !== "undefined") ? io() : null;

export const ready = new Promise((resolve) => {
  if (!socket) {
    console.warn("socket.io client not found; live progress disabled.");
    resolve(null);
    return;
  }

  socket.on("connected", (data) => {
    room = data.room;
    resolve(room);
  });

  socket.on("connect", () => {
    socket.emit("join");
  });

  socket.on("progress", (data) => {
    window.dispatchEvent(new CustomEvent("ft:progress", { detail: data }));
  });
  socket.on("done", (data) => {
    window.dispatchEvent(new CustomEvent("ft:done", { detail: data }));
  });
  socket.on("error", (data) => {
    window.dispatchEvent(new CustomEvent("ft:error", { detail: data }));
  });
});

export function getRoom() {
  return room;
}

export function emitStop() {
  if (socket) socket.emit("stop");
}

export default socket;
