// CXL transport — devdax mmap + load for L3 CXL client.
// Linux-only. No external libraries (pure syscall).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;

#ifndef L3_VERSION
#define L3_VERSION "unknown"
#endif

class CXLTransport {
public:
    CXLTransport() = default;

    void open(const std::string& path, size_t size) {
        if (base_) {
            throw std::runtime_error("CXL transport already open");
        }

        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open " + path + ": " + strerror(errno));
        }

        void* ptr = ::mmap(nullptr, size, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
        ::close(fd);

        if (ptr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap " + path + " (" +
                                     std::to_string(size) + " bytes): " + strerror(errno));
        }

        base_ = ptr;
        size_ = size;
        path_ = path;
    }

    py::bytes load(uint64_t offset, uint32_t size) {
        if (!base_) {
            throw std::runtime_error("CXL transport not open");
        }
        if (offset + size > size_) {
            throw std::runtime_error("CXL load out of bounds: offset=" +
                                     std::to_string(offset) + " size=" +
                                     std::to_string(size) + " device_size=" +
                                     std::to_string(size_));
        }

        const char* src = static_cast<const char*>(base_) + offset;

        // Release GIL for large values (>4KB)
        if (size > 4096) {
            py::gil_scoped_release release;
            std::string result(src, size);
            py::gil_scoped_acquire acquire;
            return py::bytes(result);
        }

        return py::bytes(src, size);
    }

    static bool is_available() {
        // Check if any /dev/dax* devices exist
        struct stat st;
        // Quick check for common devdax path
        if (::stat("/dev/dax0.0", &st) == 0 && S_ISCHR(st.st_mode)) {
            return true;
        }
        return false;
    }

    void close() {
        if (base_) {
            ::munmap(base_, size_);
            base_ = nullptr;
            size_ = 0;
            path_.clear();
        }
    }

    ~CXLTransport() {
        close();
    }

    bool is_open() const { return base_ != nullptr; }
    size_t device_size() const { return size_; }
    std::string device_path() const { return path_; }

private:
    void* base_ = nullptr;
    size_t size_ = 0;
    std::string path_;
};

PYBIND11_MODULE(_l3_cxl, m) {
    m.doc() = "L3 CXL transport — devdax mmap + load";

    m.attr("__version__") = L3_VERSION;

    m.def("is_available", &CXLTransport::is_available,
          "Check if CXL devdax devices are available");

    py::class_<CXLTransport>(m, "CXLTransport")
        .def(py::init<>())
        .def("open", &CXLTransport::open, py::arg("path"), py::arg("size"),
             "Open and mmap a devdax device")
        .def("load", &CXLTransport::load, py::arg("offset"), py::arg("size"),
             "Load bytes from device offset")
        .def("close", &CXLTransport::close, "Unmap the device")
        .def("is_open", &CXLTransport::is_open, "Check if device is mapped")
        .def("device_size", &CXLTransport::device_size, "Get mapped device size")
        .def("device_path", &CXLTransport::device_path, "Get device path");
}
