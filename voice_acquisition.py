# Start first program in ESP32
from os import curdir
from os import remove as remove
from os.path import join as pjoin
import datetime

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from cgi import parse_header, parse_multipart
import glob

PORT = 3000
filename1 = "raw_speech"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".raw"

class StoreHandler(BaseHTTPRequestHandler):
    STORE_PATH = './data/raw_data'


    def do_POST(self):

        file_path = pjoin(self.STORE_PATH, filename1)
        length = self.headers['content-length']
        data = self.rfile.read(int(length))
        ctype, pdict = parse_header(self.headers['content-type'])
        print("Got:", length, "ctype", ctype)
        with open(file_path, 'ab') as fh:
            fh.write(data)
        self.send_response(200)


# ************ DELETE and loop ********************************

# DEL_PATH = '/home/development/Projects/ML/model_training_esp32/data/esp32_mel'
# files = glob.glob(DEL_PATH + "/*")
# for f in files:
#    remove(f)
# print("Removed Files & Start Server, listening on:", PORT)

print("Start Server on", PORT)
server = HTTPServer(('', 3000), StoreHandler)
server.serve_forever()
