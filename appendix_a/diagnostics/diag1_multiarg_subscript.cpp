#include <mdspan/mdspan.hpp>
int main() {
    float data[12];
    auto view = std::mdspan(data, 3, 4);
    return static_cast<int>(view[1, 2]);
}
