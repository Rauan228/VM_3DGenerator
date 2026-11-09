"""Read members of a remote zip over HTTP range requests, without downloading the whole archive (Slovo is 16 GB
trimmed, 105 GB untrimmed; the 300 words need a few GB of it).

    python3 remote_zip.py list URL            -> member names and sizes
    python3 remote_zip.py get URL member out  -> extract one member
"""
import io, sys, zipfile, urllib.request


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception:
                if attempt == 4: raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))


if __name__ == "__main__":
    z = open_zip(sys.argv[2])
    if sys.argv[1] == "list":
        for i in z.infolist(): print(i.filename, i.file_size, i.compress_size)
    else:
        with z.open(sys.argv[3]) as src, open(sys.argv[4], "wb") as dst: dst.write(src.read())
