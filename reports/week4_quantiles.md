# Tuần 4 — Kết quả dự báo phân vị và tổng kết kỹ thuật

## Kết quả đã hoàn thành

Đã huấn luyện LightGBM cho q05, q50, q90, q95; chấm đủ sáu cửa sổ
validation và sáu cửa sổ test; lưu pinball loss, độ phủ, độ rộng và đảo thứ
tự phân vị. Đã xuất bốn ví dụ dự báo và phân tích độ phủ theo chuỗi, mức
bán, cửa sổ và bước dự báo. Người thực hiện đã gửi kết quả chạy và xác
nhận mục 5. Bản [tóm tắt Mốc 1](milestone1_summary.md) đã sẵn sàng để
người thực hiện gửi thầy cô; chưa có xác nhận đã gửi hoặc nhận góp ý.

## Dữ liệu, mô hình và cách đánh giá

Giữ 297 chuỗi FOODS của CA_3, TX_2, WI_1. Train ban đầu d_1–d_1605;
validation d_1606–d_1773; test d_1774–d_1941. Mỗi tập đánh giá gồm sáu
cửa sổ 28 ngày. Dự báo lại ở đầu mỗi cửa sổ, chỉ dùng số bán đã xảy ra;
lịch sử mở rộng khi chuyển cửa sổ. Test đã được dùng để báo cáo kết quả,
không còn là tập chưa từng xem và không dùng chọn lại cấu hình.

Một mô hình phân vị dùng chung cho chuỗi và bước dự báo, học riêng mỗi
mức xác suất. Có 23 đặc trưng, gồm lịch sử tại mốc, thông tin chuỗi, lịch
của ngày cần dự báo và horizon 1–28. Khác tuần 3, dự báo trực tiếp từ cùng
mốc lịch sử, không điền q50/q95 làm số bán để dự báo ngày kế tiếp. Không
quy chênh lệch giữa hai thiết kế cho riêng thay đổi hàm mất mát.

500 cây/mô hình, learning rate 0,05, tối đa 31 lá, min child samples 50,
seed 29, bốn luồng CPU là **lựa chọn kỹ thuật xuất phát**. Chưa tối ưu
các tham số này theo pinball loss. Mỗi khối nhãn train dài 28 ngày và
kết thúc không muộn hơn mốc fit; các mốc train cách nhau 28 ngày.

Kết quả chính dùng dự báo âm đặt về 0, giữ số thực, không sắp xếp lại các
phân vị. Bản raw được giữ để đối chiếu. Mỗi thước đo tính cho từng
chuỗi/cửa sổ, trung bình sáu cửa sổ trong từng chuỗi rồi trung bình đều
qua 297 chuỗi. Độ rộng dùng đơn vị hàng; coverage và crossing là tỷ lệ
quan sát. Pinball dùng đơn vị hàng có trọng số của hàm mất mát.

Giữ các **giả định**: doanh số đại diện nhu cầu, 25/12 đóng cửa, biết
trước lịch sự kiện/SNAP, giá tương lai giữ ở mức quan sát tại mốc.
Validation chấm 49.896 quan sát; test chấm 49.599 sau khi loại một ngày
đóng cửa. Tập con ban đầu đã được chọn với thông tin toàn giai đoạn;
kết luận có điều kiện trên tập con cố định đó.

## Kết quả trên validation và test

Các số sau là trung bình theo cách gộp đã mô tả, không phải số giả định.

- **Pinball validation**, theo thứ tự q05 / q50 / q90 / q95:
  **0,113422 / 0,704293 / 0,455200 / 0,295940**.
- **Pinball test**, cùng thứ tự:
  **0,118901 / 0,751754 / 0,481741 / 0,313974**.
- **Độ phủ [q05, q95]**: validation **94,869328%**, test **94,285002%**.
- **Độ rộng**: validation **6,048985**, test **6,146186 đơn vị hàng**.
- **Có ít nhất một cặp phân vị liền kề đảo thứ tự**: validation
  **4,753888%**, test **3,486585%**. Đầu mút q05 > q95 có tỷ lệ 0%.
- **Số chuỗi có độ phủ dưới 90%**: validation **15/297**, test **27/297**.

Không so độ lớn pinball giữa các mức phân vị để xếp hạng chúng; muốn so
mô hình phải giữ cùng mức phân vị, dữ liệu và cách gộp. Chưa có baseline
phân vị để kết luận mô hình khoảng tốt hơn cách thống kê lịch sử.

q05 clipped bằng 0 ở mọi dự báo, nên khoảng là [0, q95] và độ phủ trùng
tỷ lệ y <= q95. Số bán rời rạc, nhiều giá trị 0 giúp giải thích độ phủ
cao hơn 90%, nhưng không chứng minh mỗi phân vị phù hợp. Trên test,
y <= q90 đạt 89,205696%; y <= q95 đạt 94,285002%, đều hơi thấp hơn mức
mục tiêu riêng của đầu mút. Chuỗi test thấp nhất chỉ đạt 75,970018%.
Chi tiết và nguồn số ở [phân tích độ phủ](week4_coverage.md).

## Biểu đồ và nguồn thí nghiệm

Ba ví dụ có mức bán thấp/giữa/cao được chọn bằng lịch sử train. Ví dụ
thứ tư có độ phủ validation thấp, được chọn sau khi chấm để chẩn đoán.
Không coi bốn ví dụ là đại diện thống kê cho mọi chuỗi.

- [Mức bán thấp](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/train_low.png).
- [Mức bán giữa](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/train_middle.png).
- [Mức bán cao](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/train_high.png).
- [Trường hợp độ phủ thấp](quantiles/6677b9628294457fb1a3df04056875d1/plots/acc89fc3c0174e42ab547c44f0bd7c9b/validation_low_coverage.png).
- [Phân bố độ phủ và biến thiên theo cửa sổ](quantiles/coverage/6bb24de09d644f58bbaa63ab0a608ad2/coverage_diagnostics.png).

Run nguồn: huấn luyện đầu `584609c2de6a466ba50a3c3c25329566`; chấm đầu
`1ad57f2967bc4c2f957c771f75c326e1`; validation
`6677b9628294457fb1a3df04056875d1`; test
`6bb24de09d644f58bbaa63ab0a608ad2`; biểu đồ
`acc89fc3c0174e42ab547c44f0bd7c9b`.

CSV đánh giá được lưu dưới `reports/quantiles/<run-id>/`. Phân tích độ
phủ lưu ở `reports/quantiles/coverage/<test-run-id>/`, kèm SHA-256 nguồn.
MLflow dùng SQLite `mlruns/mlflow.db`. Git lưu code/cấu hình/CSV/hình;
`mlruns/` và dữ liệu xử lý bị bỏ qua, cần sao lưu riêng nếu muốn giữ
mô hình, cơ sở dữ liệu và toàn bộ lịch sử thí nghiệm.

Hướng dẫn tái tạo từng bước: [huấn luyện](../docs/week4_training.md),
[đánh giá](../docs/week4_evaluation.md), [vẽ](../docs/week4_plots.md),
[phân tích độ phủ](week4_coverage.md). Lệnh fit/backtest sẽ tạo run mới;
không cần chạy lại test để hoàn thành bản tổng kết này.

## Giới hạn và bàn giao sang tuần 5

Kết quả mới đo chất lượng dự báo; chưa trả lời câu hỏi chính về chi phí
ở cùng mức đáp ứng. Chưa mô phỏng kho, chưa chốt chi phí/lead time,
chưa có baseline phân vị và chưa kiểm định chênh lệch giữa các mô hình.
Không mặc định cộng q95 theo ngày thành q95 của tổng nhu cầu chờ hàng.

Tuần 5 sẽ xây bộ mô phỏng kho từng ngày. Trước khi viết, cần thống nhất
thứ tự nhận hàng–bán hàng–đặt hàng, cách tính thời gian chờ, xử lý thiếu
hàng và trạng thái ban đầu. Mọi tham số vận hành tự đặt phải ghi là giả
định. Đây là việc kế tiếp, chưa được triển khai trong tuần 4.

## Kiểm tra khi chốt tuần

- Toàn bộ **150 tests đạt** bằng `python -m pytest -q --tb=short -p no:cacheprovider`.
- Ruff đạt trên các file code và test tuần 4, gồm phân tích độ phủ.
- 28 liên kết tệp cục bộ trong các tài liệu được kiểm tra tồn tại; 13
  SHA-256 đầu vào trong bản ghi phân tích độ phủ khớp tệp trên máy.
- Có một cảnh báo deprecation của SQLAlchemy bên trong MLflow; không
  làm test thất bại. Không huấn luyện lại hoặc chọn lại tham số bằng test
  trong bước chốt tài liệu này.
