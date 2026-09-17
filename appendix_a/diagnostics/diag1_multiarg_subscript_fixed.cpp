#include <mdspan/mdspan.hpp>
#include <array>
std::array<size_t,2> idx2(size_t a, size_t b) { return {a, b}; }
int main() {
    float data[12];
    auto view = std::mdspan(data, 3, 4);
    return static_cast<int>(view[idx2(1, 2)]);
}
