# Mốc 1 — Tóm tắt kết quả tuần 1–4

**Đề tài:** Dự báo nhu cầu theo khoảng và tối ưu cách đặt hàng tồn kho
trên dữ liệu bán lẻ (M5).

**Trạng thái:** bản chuẩn bị để sinh viên gửi giảng viên. Chưa ghi nhận
việc gửi hoặc góp ý từ giảng viên. Các số dưới đây là kết quả thực nghiệm;
các giả định được nêu riêng. Không có kết quả mô phỏng chi phí ở mốc này.

## Bài toán và dữ liệu

Em muốn kiểm tra: ở cùng mức đáp ứng nhu cầu, cách đặt hàng dựa trên dự
báo khoảng có giảm chi phí tồn kho so với cách dựa trên thống kê lịch sử
không, và kết luận có bền khi thay đổi giả định chi phí không?

Em đã cố định 297 chuỗi sản phẩm–cửa hàng thuộc FOODS ở CA_3, TX_2, WI_1.
Train d_1–d_1605; validation d_1606–d_1773; test d_1774–d_1941.
Mỗi tập đánh giá gồm sáu cửa sổ 28 ngày, mở rộng lịch sử sau từng cửa sổ.
Thước đo tính từng chuỗi/cửa sổ rồi trung bình đều qua cửa sổ và chuỗi.

**Giả định:** doanh số đại diện cho nhu cầu; 25/12 đóng cửa; lịch sự kiện
và SNAP biết trước; giá tương lai giữ ở mức đã biết tại mốc. M5 không cho
biết nhu cầu mất do hết hàng. Tập con ban đầu đã được chọn bằng thông tin
toàn giai đoạn, nên kết luận chỉ áp dụng có điều kiện trên tập con này.

## Dự báo điểm: baseline và LightGBM

Đã thử bảy baseline. Trên cùng sáu cửa sổ validation, RMSSE trung bình
(càng thấp càng tốt) của ETS là **0,673912**, trung bình trượt 28 ngày
**0,677792**, LightGBM 500 cây **0,694558**. LightGBM xếp thứ ba, chưa
vượt hai baseline mạnh nhất. Trên test, ETS đạt **0,721437**, trung bình
trượt đạt **0,727876**; LightGBM dự báo điểm tuần 3 chưa được chấm test.
Không dùng điểm test của baseline để khẳng định thứ hạng của LightGBM.

## Dự báo phân vị và khoảng

Đã học q05, q50, q90, q95 bằng LightGBM, dùng thiết kế dự báo trực tiếp
1–28 ngày từ cùng mốc lịch sử, 23 đặc trưng và 500 cây cho mỗi mô hình.
Đây là cấu hình xuất phát, chưa tối ưu theo pinball loss. Kết quả chính
đặt dự báo âm về 0 và giữ cả bản gốc; không sắp xếp lại phân vị.

- Khoảng danh nghĩa 90% `[q05, q95]` phủ **94,87% trên validation** và
  **94,29% trên test**; độ rộng trung bình **6,05** và **6,15 đơn vị hàng**.
- Pinball loss q05/q50/q90/q95 trên test lần lượt **0,118901 / 0,751754 /
  0,481741 / 0,313974**. Chưa có baseline phân vị để kết luận ưu thế.
- **27/297 chuỗi test** có độ phủ dưới 90%; chuỗi thấp nhất khoảng
  **75,97%**. q05 clipped đều bằng 0, nên số bán rời rạc và nhiều ngày
  bán 0 giúp giải thích độ phủ gộp cao hơn mức danh nghĩa.
- Tỷ lệ có phân vị đảo thứ tự là **4,75% validation**, **3,49% test**;
  q05 > q95 bằng 0%. Đã lưu chẩn đoán theo chuỗi, cửa sổ và bước dự báo.

Độ phủ khoảng chưa phải tỷ lệ lượng hàng được đáp ứng. Chưa có cơ sở để
kết luận giảm chi phí tồn kho. Không mặc định cộng các phân vị từng ngày
thành phân vị tổng nhu cầu trong thời gian chờ hàng.

## Kế hoạch tiếp theo và nội dung xin góp ý

Tuần 5 xây bộ mô phỏng tồn kho theo ngày, sau đó so sánh ba cách đặt
hàng: ngưỡng cố định, (s,S) theo thống kê lịch sử và theo dự báo khoảng.
Chi phí, thời gian chờ và trạng thái kho ban đầu sẽ là **giả định mô
phỏng**, được ghi rõ và kiểm tra độ nhạy. PyTorch là phần tùy chọn.

Em xin góp ý về: phạm vi 297 chuỗi và cách đánh giá hiện tại; việc giữ ETS
và thống kê lịch sử làm mốc đối chiếu; định nghĩa mức đáp ứng và cách so
sánh chi phí ở cùng mức đáp ứng; các giả định tối thiểu nên có trong mô
phỏng. Em cũng muốn xác nhận mức ưu tiên xử lý đảo thứ tự phân vị trước
khi ghép dự báo vào cách đặt hàng.

## Tài liệu kèm theo

- [Baseline và dữ liệu](week2_baselines.md), [LightGBM dự báo điểm](week3_lightgbm.md).
- [Báo cáo tuần 4](week4_quantiles.md), [phân tích độ phủ](week4_coverage.md).
- [Ví dụ dự báo và khoảng](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/train_middle.png),
  [trường hợp độ phủ thấp](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/validation_low_coverage.png).

Có thể dùng phần nội dung trên làm bản gửi thầy cô; nếu gửi tệp Markdown,
cần gửi kèm hình theo các đường dẫn tương đối hoặc xuất cùng hình để
người nhận xem được. Không gửi riêng thư mục MLflow thay cho báo cáo.
