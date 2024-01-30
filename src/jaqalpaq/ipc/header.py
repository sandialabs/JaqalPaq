import struct
from jaqalpaq.error import JaqalError


class IPCHeader:
    """Represent information in the IPC protocol's header."""

    version: int = 0

    _encoding_str = "!BI"
    _encoding_size = struct.calcsize(_encoding_str)

    @classmethod
    def from_body(cls, body):
        if not isinstance(body, bytes):
            raise JaqalError("Body must be converted to bytes")
        size = len(body)
        return cls(size)

    @classmethod
    def recv(cls, sock):
        """Read this header from a socket. If there was no data
        remaining on the socket, return None; for all other
        exceptional cases raise an exception."""
        raw = sock.recv(cls._encoding_size)
        if len(raw) != cls._encoding_size:
            if len(raw) == 0:
                return None
            raise JaqalError(f"Could not read header, got {raw}")
        verbyte, size = struct.unpack(cls._encoding_str, raw)
        if verbyte != cls.version_byte:
            raise JaqalError(
                f"Version byte mismatch, expected 0x{cls.version_byte:x}, found 0x{verbyte:x}"
            )
        return cls(size)

    def __init__(self, size):
        self.size = size

    @classmethod
    @property
    def version_byte(cls):
        """Return the version byte as an integer."""
        return 0x80 | cls.version

    def send(self, sock):
        """Write this header to a socket."""
        data = struct.pack(self._encoding_str, self.version_byte, self.size)
        count = sock.send(data)
        if count != self._encoding_size:
            raise JaqalError("Could not send header")
