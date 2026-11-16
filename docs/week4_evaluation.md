# Tuần 4 — Mục 3A: chấm dự báo phân vị trên cửa sổ đầu

Run dự báo nguồn đã xác nhận: `584609c2de6a466ba50a3c3c25329566`.
Mục 3A đã hoàn thành. Người thực hiện đã gửi kết quả run chấm
`1ad57f2967bc4c2f957c771f75c326e1`; trạng thái và ba CSV đã được đối chiếu
với artifact MLflow. Không huấn luyện lại ở bước này.

## Những gì được chấm

Cửa sổ d_1606–d_1633, 297 chuỗi × 28 ngày = 8.316 quan sát. Nhãn thật
chỉ được lấy để chấm dự báo đã lưu, không đưa trở lại mô hình. Quy tắc
loại 25/12 được giữ; cửa sổ đầu không có ngày này nên dự kiến loại 0 ngày.

Hai biến thể được lưu song song:

- `clipped`: dự báo đã đặt số âm về 0 từ mục 2; đây là kết quả chính.
- `raw`: dự báo gốc, dùng để đối chiếu ảnh hưởng của quy tắc trên.

Không thay mô hình hoặc sắp xếp lại phân vị trong bước chấm. Script kiểm
tra khóa đủ/không trùng, mốc, horizon, ngày lịch, tập chuỗi và tính nhất quán
`clipped = max(raw, 0)`. Cấu hình chia dữ liệu/đóng cửa phải khớp bản lưu
trong run nguồn. Lệnh này chỉ chấm cửa sổ validation đầu, không chấm test.

Trước lần chấm, đã kiểm tra bộ đầu vào thật của run nguồn: các kiểm tra trên
đều đạt cho 8.316 dòng và 297 chuỗi. Đã xử lý
khác biệt độ phân giải ngày giữa CSV (ns) và parquet (us), kèm test hồi quy.
Bộ kiểm tra toàn repo: 141 tests đạt; Ruff đạt trên các file tuần 4. Còn
cảnh báo deprecation SQLAlchemy bên trong MLflow như trước.

## Đọc các thước đo

- `pinball_q05/q50/q90/q95`: loss riêng từng phân vị, càng thấp càng tốt
  khi so cùng phân vị và dữ liệu. Không phải phần trăm hoặc chi phí tồn kho.
  Công thức theo [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss).
- `coverage_le_q90`, `coverage_le_q95`: tỷ lệ số ngày `y <= q90/q95`.
  Đây là độ phủ một phía, không phải độ phủ khoảng [q05, q95].
- `coverage_lt_q...`: tỷ lệ `y < q`, lưu thêm trong CSV. So với phép `<=`
  giúp nhìn ảnh hưởng của các giá trị trùng đầu mút trên dữ liệu số nguyên.
- `interval90_coverage`: tỷ lệ ngày `q05 <= y <= q95`, tính cả đầu mút.
  Mức đặt mục tiêu là 90%; kết quả đo không mặc định bằng mức danh nghĩa.
- `interval90_mean_width_ordered`: độ rộng trung bình `q95 - q05` trên
  các khoảng có hai đầu đúng thứ tự; đơn vị là số lượng hàng.
- `crossing_rate`: tỷ lệ ngày có ít nhất một cặp phân vị kề nhau đảo thứ
  tự, ví dụ q90 > q95. CSV lưu thêm tỷ lệ của từng cặp.

Cách lấy khoảng từ q05/q95 và tính coverage có trong
[ví dụ chính thức scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).
Các quy tắc xử lý khoảng sai thứ tự dưới đây là **quy ước đánh giá của
dự án**, không phải tiêu chuẩn nghiệp vụ hoặc bảo đảm từ thư viện.

## Nếu phân vị bị đảo thứ tự

Một ngày có q50 > q90 vẫn có thể có q05 <= q95. Vì vậy cần đo cả
`crossing_rate` của bốn phân vị và `interval90_invalid_rate` của hai đầu khoảng.

Với ngày có q05 > q95:

- Tính ngày đó là **không được phủ**, không loại nó khỏi mẫu số coverage.
- Không dùng độ rộng âm; độ rộng của ngày đó là không xác định.
- `n_ordered_interval` ghi số ngày còn đủ điều kiện tính độ rộng.
- `interval90_below_rate` và `interval90_above_rate` chỉ tính sự kiện nằm
  dưới/trên khoảng có thứ tự đúng, nhưng mẫu số vẫn là mọi ngày được chấm.

Bốn tỷ lệ coverage + below + above + invalid cộng thành 1. Nếu một chuỗi
không còn khoảng đúng thứ tự, độ rộng của chuỗi là NaN và số chuỗi xác định
được thước đo được ghi rõ. Không thay NaN thành 0 hoặc giấu trường hợp này.

## Cách gộp kết quả

1. Tính từng thước đo trên các ngày được chấm của **từng chuỗi**.
2. Lấy trung bình đều và trung vị qua 297 chuỗi, theo quy tắc đã dùng từ
   tuần 2. Pinball loss vẫn tính theo đơn vị hàng, chưa chuẩn hóa quy mô.
3. Lưu thêm kết quả riêng từng bước dự báo 1–28, gộp qua các chuỗi tại bước
   đó. Nếu một bước là ngày đóng cửa, số quan sát là 0, thước đo là NaN.

Số 90% là mức phủ danh nghĩa, không phải một ngưỡng đạt/trượt tuyệt đối.
Số bán rời rạc, nhiều số 0, số ngày hữu hạn và các chuỗi phụ thuộc nhau
cần được xem khi nhận xét. Chưa đặt ngưỡng sai lệch coverage được chấp nhận.
Không suy ra fill rate hoặc chi phí tồn kho từ các kết quả này.

## Lệnh chạy

Từ thư mục gốc repo, khi `.venv` đã được kích hoạt:

```powershell
python -m src.evaluation.score_quantiles --run-id 584609c2de6a466ba50a3c3c25329566
```

Script đọc dự báo trong artifact MLflow, ghi một **run chấm điểm mới**
trong experiment `week4-quantile-forecasting`, liên kết bằng tag
`source_forecast_run_id`. Run huấn luyện cũ giữ nguyên thông tin ban đầu;
`validation_scored=false` ở run cũ là đúng vì nó chỉ huấn luyện/xuất dự báo.

Console in pinball loss, độ phủ khoảng, độ rộng và crossing của bản clipped.
**Console in tỷ lệ theo phần trăm; CSV và MLflow lưu tỷ lệ trong [0, 1].**
Các số đã được xác nhận của lần chạy này ghi ở phần kết quả dưới đây;
chạy lại sẽ tạo một run chấm điểm mới.

## File kết quả

Thư mục mới:
`reports/quantiles/<run-nguon>/scoring/<run-cham-diem>/`.

- `per_series_window.csv`: từng chuỗi × biến thể, gồm số quan sát và số
  khoảng hợp lệ. Cửa sổ này có 594 dòng = 297 chuỗi × 2 biến thể.
- `per_horizon_window.csv`: từng bước dự báo × biến thể, 56 dòng.
- `summary_window.csv`: trung bình, trung vị và số chuỗi xác định cho
  từng thước đo × biến thể.

MLflow lưu các CSV, mã nguồn chấm điểm, quy tắc gộp và bản ghi nguồn với
SHA-256 để nhận diện các file đã chấm. Không cần chạy lại lệnh huấn luyện.

## Kết quả đã xác nhận của mục 3A

Run chấm `1ad57f2967bc4c2f957c771f75c326e1` hoàn tất, nguồn dự báo
`584609c2de6a466ba50a3c3c25329566`. Đủ 8.316 quan sát ở 297 chuỗi,
không loại ngày đóng cửa; cả 297 chuỗi có giá trị cho các thước đo.
Ba CSV cục bộ trùng từng byte với bản lưu trong artifact của run chấm.

Kết quả clipped, mỗi cặp dưới đây là trung bình / trung vị qua các chuỗi:

- Pinball q05: 0,116811 / 0,051786.
- Pinball q50: 0,671014 / 0,458905.
- Pinball q90: 0,434244 / 0,279497.
- Pinball q95: 0,276341 / 0,180076.
- Độ phủ [q05, q95]: 95,658971% / 96,428571%.
- Độ phủ một phía q90: 91,221741% / 92,857143%.
- Độ phủ một phía q95: 95,658971% / 96,428571%.
- Độ rộng khoảng: 6,075209 / 3,665450 đơn vị hàng.
- Crossing qua bốn phân vị: 3,799904% / 0%.
- Khoảng có q05 > q95: 0% / 0%.

Các số này lấy từ `summary_window.csv`, không phải số giả định. Mỗi chuỗi
có đúng 28 ngày được chấm nên độ phủ trung bình cũng bằng 7.955 / 8.316;
có 361 quan sát vượt đầu trên và không có quan sát dưới đầu dưới.

### Vì sao độ phủ cao hơn 90%?

Kiểm tra trực tiếp file dự báo cho thấy q05 bằng 0 ở **toàn bộ 8.316 dòng**,
cả bản gốc lẫn clipped. Vì doanh số không âm, khoảng hiện là [0, q95],
không có phần bị loại ở phía dưới. Do đó độ phủ khoảng trùng độ phủ một
phía của q95: 95,658971%. Đây là diễn giải từ dự báo đã lưu, không phải
bằng chứng rằng khoảng đã được hiệu chỉnh tốt cho mọi chuỗi.

Dữ liệu rời rạc/nhiều số 0 có thể khiến một số phân vị cùng bằng 0, như
đã giải thích ở mục 1. Tuy nhiên, không mặc định q05 = 0 phù hợp mọi sản
phẩm: trong cửa sổ này có 23/297 chuỗi có tỷ lệ số bán <= q05 dưới 5%,
trong đó 12 chuỗi không có ngày nào như vậy. Mỗi chuỗi chỉ có 28 quan sát;
cần xem các cửa sổ khác trước khi kết luận về độ phù hợp của q05.

Không quy riêng độ phủ cao cho việc khoảng quá rộng; cần nhìn đồng thời
độ rộng, tính rời rạc và kết quả từng chuỗi. Chưa có baseline phân vị để
so pinball loss nên các số loss trên chưa chứng minh mô hình tốt hơn
một cách dự báo xác suất khác.

### Crossing và ảnh hưởng của đặt âm về 0

Clipped có 316/8.316 quan sát crossing, xuất hiện ở 57/297 chuỗi:

- q05 > q50: 0 quan sát.
- q50 > q90: 59 quan sát.
- q90 > q95: 257 quan sát.

Không có đầu mút q05 > q95 ở bản clipped; vì vậy khoảng [q05, q95] vẫn
có thứ tự đúng dù một số phân vị bên trong bộ bốn bị đảo. Trung vị crossing
bằng 0 không có nghĩa toàn bộ dự báo đã có thứ tự đúng.

Bản raw có 530/8.316 quan sát crossing (6,373256%) và 7 khoảng q05 > q95.
Đặt âm về 0 giảm crossing xuống 316 quan sát, giảm số khoảng sai đầu mút
về 0; coverage tăng từ 95,574796% lên 95,658971%, tức thêm 7 quan sát
được phủ. Quy tắc đặt âm về 0 đã chọn trước lần chấm, không chọn theo kết
quả này. Chưa áp dụng sắp xếp hoặc hiệu chỉnh phân vị để sửa crossing.

## Phần còn lại của mục 3

Mục 3A đã được xác nhận. Kết quả mới thuộc cửa sổ đầu, chưa đại diện cho
cả sáu cửa sổ validation và chưa hoàn thành toàn bộ mục 3 của tuần 4.
Các bước tiếp theo:

- 3B: đánh giá đủ sáu cửa sổ validation với quy tắc refit/giới hạn lịch sử
  đã thống nhất, rồi gộp theo chuỗi/cửa sổ.
- 3C: chỉ đánh giá test sau khi khóa cấu hình và quy tắc dự báo/chấm điểm.
  Test không được dùng để chọn hoặc chỉnh lại mô hình.
