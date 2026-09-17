#include <mdspan>
int main() {
    float data[12];
    std::mdspan view(data, 3, 4);
    return 0;
}
