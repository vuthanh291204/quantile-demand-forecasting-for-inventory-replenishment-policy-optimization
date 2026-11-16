# Tuần 4 — Đánh giá dự báo phân vị

Run dự báo nguồn đã xác nhận: `584609c2de6a466ba50a3c3c25329566`.
Mục 3A đã hoàn thành. Người thực hiện đã gửi kết quả run chấm
`1ad57f2967bc4c2f957c771f75c326e1`; trạng thái và ba CSV đã được đối chiếu
với artifact MLflow. Không huấn luyện lại ở bước này.

Mục 3B đã hoàn thành với run `6677b9628294457fb1a3df04056875d1` do người
thực hiện gửi. Đã đối chiếu các file kết quả với artifact MLflow.
Mục 4 thực hiện theo [hướng dẫn biểu đồ](week4_plots.md).

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

## Mục 3B đánh giá đủ sáu cửa sổ validation

Script `src/evaluation/backtest_quantiles.py` đọc cấu hình hiệu lực từ run
huấn luyện `584609c2de6a466ba50a3c3c25329566`. Các lựa chọn kỹ thuật xuất
phát gồm q05/q50/q90/q95, 500 cây, 23 đặc trưng và khoảng cách mốc train
28 ngày. Những lựa chọn này được ghi cùng kết quả trong MLflow.

Lần chạy 3B huấn luyện lại cả bốn mô hình tại mỗi mốc, tổng cộng 24 mô hình.
Các cửa sổ dựa trên `configs/split.toml` đã lưu trong run nguồn.

Kiểm tra toàn dự án đạt 146 tests, gồm luồng sáu cửa sổ trên dữ liệu giả.
Ruff đạt trên các file thay đổi. Bước chuẩn bị bảng M5 đã được kiểm tra
đủ sáu mốc, với số dòng train tăng từ 387.856 lên 429.436 và mỗi cửa sổ
có 8.316 dòng đầu vào dự báo. Kết quả lần chạy M5 đã được xác nhận ở phần
dưới đây.

- Cửa sổ 1 dùng lịch sử đến d_1605, dự báo d_1606–d_1633.
- Cửa sổ 2 dùng lịch sử đến d_1633, dự báo d_1634–d_1661.
- Cửa sổ 3 dùng lịch sử đến d_1661, dự báo d_1662–d_1689.
- Cửa sổ 4 dùng lịch sử đến d_1689, dự báo d_1690–d_1717.
- Cửa sổ 5 dùng lịch sử đến d_1717, dự báo d_1718–d_1745.
- Cửa sổ 6 dùng lịch sử đến d_1745, dự báo d_1746–d_1773.

Khi chuyển sang cửa sổ 2, doanh số đã xảy ra ở cửa sổ 1 trở thành lịch sử
để học và tạo đặc trưng. Cách mở rộng lịch sử này áp dụng lần lượt cho
các cửa sổ sau, cùng lịch refit của baseline tuần 2 và LightGBM tuần 3.
Trong mỗi cửa sổ, các đầu vào lịch sử được cố định tại mốc dự báo.

### Gộp kết quả qua các cửa sổ

1. Tính các thước đo riêng cho từng cặp chuỗi/cửa sổ, cho cả raw và clipped.
2. Lấy trung bình sáu cửa sổ cho từng chuỗi.
3. Lấy trung bình đều và trung vị qua 297 chuỗi để báo cáo kết quả chung.

Các bộ đếm `n_evaluated` và `n_ordered_interval` được cộng qua cửa sổ.
`n_windows` ghi số cửa sổ của mỗi chuỗi. `width_windows_defined` ghi số
cửa sổ tính được độ rộng. Một thước đo gộp của chuỗi được xác định khi
có giá trị ở đủ sáu cửa sổ. `n_series_defined` cho biết số chuỗi có giá
trị để tính trung bình và trung vị của thước đo đó.

### Chạy mục 3B

Tại thư mục gốc dự án, trong terminal đang kích hoạt `.venv`, chạy lệnh sau.

```powershell
python -m src.evaluation.backtest_quantiles --source-run-id 584609c2de6a466ba50a3c3c25329566
```

Console sẽ in tiến độ từ `Window 1/6` đến `Window 6/6`, số vòng của mỗi
mô hình và bảng chỉ số tổng hợp. Lần chạy tạo một run mới trong experiment
`week4-quantile-forecasting`. Thời gian chạy M5 sẽ được quan sát từ lần
thực hiện của người dùng.

Khi hoàn tất, đối chiếu các dòng sau với kích thước suy ra từ thiết kế.
49.896 bằng 297 chuỗi × 6 cửa sổ × 28 ngày.

```text
Validation windows: 6; series: 297
Forecast shape: (49896, 14)
Forecast days: 1606 1773
Evaluated observations: 49896
Status: FINISHED
```

Gửi output từ phần `Window 6/6` đến cuối, gồm các chỉ số tổng hợp, Run ID
và đường dẫn `Saved results`. Nếu gặp lỗi, gửi traceback để xử lý.

### Các file của mục 3B

Thư mục kết quả là `reports/quantiles/<run-3b>/`. MLflow lưu cùng các CSV
và mô hình theo đường dẫn `models/window_1/` đến `models/window_6/`.

- `forecasts_valid.csv` chứa 49.896 dự báo với cột `window` để truy vết.
- `per_window_valid.csv` chứa 3.564 dòng chuỗi/cửa sổ/biến thể.
- `per_series_valid.csv` chứa 594 dòng chuỗi/biến thể sau khi gộp cửa sổ.
- `per_horizon_valid.csv` chứa 336 dòng bước dự báo/cửa sổ/biến thể.
- `summary_valid.csv` ghi trung bình, trung vị và số chuỗi xác định.
- `window_info_valid.csv` ghi mốc, khoảng dự báo, số dòng train, số mốc
  train và ngày nhãn train cuối cùng của từng cửa sổ.

Console hiển thị tỷ lệ bằng phần trăm. Các CSV và metrics MLflow lưu tỷ
lệ trong [0, 1], theo quy ước ở mục 3A.

### Kết quả đã xác nhận của mục 3B

Run `6677b9628294457fb1a3df04056875d1` hoàn tất, đủ sáu cửa sổ và 24
mô hình, mỗi mô hình học 500 vòng. Có 49.896 dự báo ở d_1606–d_1773,
khóa chuỗi/ngày đầy đủ và duy nhất. CSV dự báo, điểm từng chuỗi, bảng tổng
hợp và thông tin từng cửa sổ trùng với bản lưu trong MLflow.

Kết quả clipped, mỗi cặp là trung bình / trung vị qua các chuỗi sau khi
lấy trung bình sáu cửa sổ cho mỗi chuỗi.

- Pinball q05 đạt 0,113422 / 0,051190.
- Pinball q50 đạt 0,704293 / 0,449566.
- Pinball q90 đạt 0,455200 / 0,299263.
- Pinball q95 đạt 0,295940 / 0,200463.
- Độ phủ [q05, q95] đạt 94,869328% / 95,238095%.
- Độ phủ một phía q90 đạt 90,510261% / 90,476190%.
- Độ rộng khoảng đạt 6,048985 / 3,442369 đơn vị hàng.
- Crossing đạt 4,753888% / 0,595238%.
- Tỷ lệ q05 > q95 bằng 0% ở bản clipped.

Cả 297 chuỗi có giá trị cho các thước đo trên. q05 bằng 0 ở toàn bộ
49.896 dòng dự báo, nên độ phủ khoảng trùng độ phủ một phía q95. Có
47.336 quan sát được phủ và 2.372 quan sát crossing. Các số được lấy từ
CSV đã lưu, đối chiếu với output người thực hiện gửi.

## Mục 3C đánh giá trên test

Sau khi hoàn thành validation và xuất biểu đồ, quay lại 3C để hoàn tất
mục lớn 3. Chốt dùng cấu hình và các quy tắc đã đánh giá trong run
`6677b9628294457fb1a3df04056875d1` cho lần test này. Script đọc cấu hình
hiệu lực từ artifact của run đó, gồm bốn phân vị, 500 cây, seed 29,
23 đặc trưng, khoảng cách mốc train 28 ngày và quy tắc đặt âm về 0.

Kiểm tra toàn dự án đạt 148 tests, gồm luồng test lấy cấu hình từ run
validation và loại một ngày đóng cửa trên dữ liệu giả. Đã kiểm tra bảng
đầu vào M5 ở cả sáu mốc test. Bảng train đầu/cuối có 437.752 / 479.332
dòng, mỗi cửa sổ có 8.316 dòng đầu vào dự báo. Lần chạy test M5 đã được
người thực hiện gửi và xác nhận ở phần kết quả cuối tài liệu.

### Cửa sổ test và lịch sử được phép

- Cửa sổ 1 dùng lịch sử đến d_1773, dự báo d_1774–d_1801.
- Cửa sổ 2 dùng lịch sử đến d_1801, dự báo d_1802–d_1829.
- Cửa sổ 3 dùng lịch sử đến d_1829, dự báo d_1830–d_1857.
- Cửa sổ 4 dùng lịch sử đến d_1857, dự báo d_1858–d_1885.
- Cửa sổ 5 dùng lịch sử đến d_1885, dự báo d_1886–d_1913.
- Cửa sổ 6 dùng lịch sử đến d_1913, dự báo d_1914–d_1941.

Ở mốc đầu, toàn bộ train và validation đã trở thành quá khứ để huấn luyện.
Khi qua cửa sổ tiếp theo, số bán của cửa sổ vừa xảy ra trở thành lịch sử
quan sát. Đây là lịch đánh giá rolling-origin đã dùng từ baseline tuần 2.
Trong mỗi cửa sổ, mô hình dùng thông tin tại mốc để dự báo 28 ngày.
Các tham số và quy tắc đã chốt được áp dụng cho cả sáu cửa sổ.

### Ngày đóng cửa và cách gộp

Test ở 07/12/2015–22/05/2016 có ngày 25/12/2015, tức d_1792 trong cửa sổ
đầu. Theo giả định đóng cửa đã dùng, ngày này bị loại khỏi thước đo.
Vì vậy có 49.896 dự báo và 49.599 quan sát được chấm, bằng 297 × 167.
Cửa sổ đầu có 27 ngày được chấm cho mỗi chuỗi, các cửa sổ sau có 28 ngày.

Tính thước đo từng chuỗi/cửa sổ, lấy trung bình đều qua sáu cửa sổ cho
mỗi chuỗi, rồi báo trung bình và trung vị qua 297 chuỗi. Vì số ngày được
chấm giữa các cửa sổ có khác nhau, độ phủ trung bình theo quy tắc này có
thể khác tỷ lệ tính gộp một lần trên toàn bộ 49.599 quan sát.

### Lệnh chạy 3C

Từ thư mục gốc, trong terminal đang kích hoạt `.venv`, chạy lệnh sau.

```powershell
python -m src.evaluation.backtest_quantiles --split test --source-run-id 6677b9628294457fb1a3df04056875d1
```

Lệnh này huấn luyện 24 mô hình và ghi kết quả test của cấu hình đã chốt.
Kết quả test dùng để đánh giá cuối, việc lựa chọn tham số dựa trên validation.

Các dòng cuối khi hoàn tất cần khớp các kích thước sau.

```text
Test windows: 6; series: 297
Forecast shape: (49896, 14)
Forecast days: 1774 1941
Evaluated observations: 49599
Closed days excluded: 1
Status: FINISHED
Test evaluated: true
```

Các file được lưu trong `reports/quantiles/<test-run-id>/`, gồm
`forecasts_test.csv`, `per_window_test.csv`, `per_series_test.csv`,
`per_horizon_test.csv`, `summary_test.csv` và `window_info_test.csv`.
MLflow liên kết run test với run validation qua tag `source_validation_run_id`.

### Kết quả đã xác nhận của mục 3C

Run `6bb24de09d644f58bbaa63ab0a608ad2` hoàn tất và liên kết với run
validation `6677b9628294457fb1a3df04056875d1`. Có đủ 24 mô hình, mỗi mô
hình 500 vòng, 49.896 dự báo và 49.599 quan sát được chấm. Loại một ngày
25/12 như dự kiến. File dự báo, điểm từng chuỗi, bảng tổng hợp và thông
tin cửa sổ đã được đối chiếu với artifact MLflow, khóa chuỗi/ngày duy nhất.

Kết quả clipped, mỗi cặp là trung bình / trung vị qua các chuỗi sau khi
lấy trung bình sáu cửa sổ cho mỗi chuỗi.

- Pinball q05 đạt 0,118901 / 0,056570.
- Pinball q50 đạt 0,751754 / 0,498223.
- Pinball q90 đạt 0,481741 / 0,326390.
- Pinball q95 đạt 0,313974 / 0,209034.
- Độ phủ [q05, q95] đạt 94,285002% / 94,620811%.
- Độ phủ một phía q90 đạt 89,205696% / 89,241623%.
- Độ rộng khoảng đạt 6,146186 / 3,844352 đơn vị hàng.
- Crossing đạt 3,486585% / 0%.
- Tỷ lệ q05 > q95 bằng 0% ở bản clipped.

Cả 297 chuỗi có giá trị cho các thước đo trên. q05 bằng 0 ở toàn bộ
49.896 dự báo test, nên độ phủ khoảng trùng độ phủ một phía q95.
Các số này là kết quả trên dữ liệu, không phải giả định vận hành.

Mục lớn 3 đã hoàn tất cùng các file kết quả validation/test. Mục 4 đã
xuất đủ bốn hình. Tiếp tục mục 5 để nhận xét độ phủ và các giới hạn.
