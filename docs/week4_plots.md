# Tuần 4 — Mục 4, biểu đồ số bán và khoảng dự báo

Dùng dự báo validation từ run `6677b9628294457fb1a3df04056875d1`, gồm
297 chuỗi qua sáu cửa sổ ở d_1606–d_1773. Script xuất ảnh PNG để dùng
trong báo cáo, đồng thời lưu dữ liệu đã vẽ và cách chọn chuỗi.

Đã tạo và kiểm tra trực quan bốn bản xem thử trên dữ liệu M5. Bộ kiểm tra
toàn dự án đạt 148 tests. Lần chạy chính thức đã hoàn thành, ghi ở cuối
tài liệu này.

## Cách chọn bốn chuỗi

Tính số bán trung bình trên train, từ lần bán dương đầu tiên đến d_1605,
loại ngày 25/12 theo giả định đóng cửa. Sắp xếp các chuỗi theo mức trung
bình này và chọn các vị trí gần 10%, 50%, 90%. Các tỷ lệ này dùng để chọn
vị trí trong danh sách chuỗi theo mức bán lịch sử.

Thêm một chuỗi có độ phủ validation thấp nhất trong các chuỗi còn lại.
Đây là trường hợp chẩn đoán được chọn sau khi xem kết quả validation,
được ghi rõ trong tiêu đề ảnh và `selected_series.csv`.

Các chuỗi được chọn trên dữ liệu hiện tại gồm các trường hợp sau.

- `FOODS_3_778_TX_2_evaluation` ở vị trí mức bán thấp theo train.
- `FOODS_2_148_WI_1_evaluation` ở vị trí giữa theo train.
- `FOODS_1_086_CA_3_evaluation` ở vị trí mức bán cao theo train.
- `FOODS_3_797_WI_1_evaluation` là trường hợp chẩn đoán độ phủ thấp.

Ba mức bán đầu dùng số liệu train để chọn. Trường hợp thứ tư dùng kết
quả validation để tìm ví dụ còn hạn chế. Các chỉ số chung của 297 chuỗi
được lấy từ mục 3B.

## Cách đọc hình

Mỗi PNG có hai phần. Phần trên vẽ toàn bộ 168 ngày validation, phần dưới
phóng to 28 ngày đầu. Các thành phần hình được thể hiện như sau.

- Đường đen và chấm đen biểu diễn số bán thực tế theo ngày.
- Đường xanh đậm biểu diễn q50, trung vị dự báo.
- Vùng xanh nhạt nằm giữa q05 và q95, có mức phủ danh nghĩa 90%.
- Đường tím chấm biểu diễn q90, giúp quan sát thứ tự các phân vị.
- Tam giác đỏ đánh dấu ngày số bán nằm ngoài khoảng.
- Vạch đứt dọc đánh dấu thời điểm chuyển cửa sổ dự báo 28 ngày.

Các đường dự báo được vẽ riêng theo từng cửa sổ. Độ phủ, độ rộng và
crossing ghi phía trên hình dùng kết quả đã tính cho chuỗi ở mục 3B.
Ảnh giữ các giá trị dự báo clipped đã lưu, gồm cả các ngày có crossing.

Ở chuỗi `FOODS_3_797_WI_1_evaluation`, bản xem thử có các đoạn q90 cao
hơn q95 và nhiều ngày số bán vượt đầu trên. Những quan sát này sẽ được
nhận xét cùng kết quả của các chuỗi khác ở bước phân tích tiếp theo.

## Chạy lệnh xuất ảnh

Từ thư mục gốc dự án, trong terminal đang kích hoạt `.venv`, chạy lệnh sau.

```powershell
python -m src.evaluation.plot_quantiles --run-id 6677b9628294457fb1a3df04056875d1
```

Script đọc artifact của run nguồn, ghép số bán thật theo khóa chuỗi/ngày,
tạo một run mới trong experiment `week4-quantile-forecasting`. Các dòng
cuối khi chạy thành công có số lượng ảnh và đường dẫn thư mục kết quả.

```text
Selected series: 4
Figures: 4
Status: FINISHED
```

Thư mục kết quả có dạng
`reports/quantiles/6677b9628294457fb1a3df04056875d1/plots/<plot-run-id>/`.
Trong đó có các file sau.

- `train_low.png`, `train_middle.png`, `train_high.png` và
  `validation_low_coverage.png` là bốn biểu đồ.
- `selected_series.csv` ghi mã chuỗi, mức bán train, chỉ số validation
  và căn cứ chọn chuỗi.
- `plot_data.csv` ghi 672 dòng đã vẽ, gồm số bán thật và các dự báo đã lưu.

Gửi output cuối lệnh, gồm `Plot run ID` và `Saved figures`, cùng xác nhận
bạn đã mở được bốn ảnh. Sau đó cùng nhận xét các hình trước khi chuyển
sang đầu mục tiếp theo.

## Kết quả xuất ảnh đã xác nhận

Run `acc89fc3c0174e42ab547c44f0bd7c9b` hoàn tất và liên kết đúng với run
validation `6677b9628294457fb1a3df04056875d1`. Người thực hiện đã gửi
output lệnh. Đã kiểm tra có đủ bốn PNG, mỗi ảnh 2240 × 1152 pixel, cùng
hai CSV trong thư mục kết quả.

Mục 4 đã xuất ảnh thành công. Phần test 3C cũng đã hoàn tất và được ghi
trong `docs/week4_evaluation.md`. Tiếp tục mục 5 để nhận xét độ phủ.
