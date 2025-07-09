# Makefile

CXX        := g++
PKG_CONFIG := pkg-config

GST_CFLAGS := $(shell $(PKG_CONFIG) --cflags gstreamer-1.0 gstreamer-rtsp-server-1.0)
GST_LIBS   := $(shell $(PKG_CONFIG) --libs   gstreamer-1.0 gstreamer-rtsp-server-1.0)
GST_LIBS   += -lgstapp-1.0      # ← 여기 추가!

CXXFLAGS   := -std=c++17 -Wall -O2

.PHONY: all clean

all: rtsp_server

rtsp_server: rtsp_server.cpp
	$(CXX) $(CXXFLAGS) $(GST_CFLAGS) rtsp_server.cpp -o rtsp_server $(GST_LIBS)

clean:
	rm -f rtsp_server