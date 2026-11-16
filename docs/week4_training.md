# Tuần 4 — Mục 2: huấn luyện LightGBM phân vị

Mục 1 đã hoàn thành: người thực hiện trả lời đúng bài tập phân vị, pinball
loss và độ phủ. Mục 2 đã hoàn thành: lần chạy M5 được người thực hiện gửi
và xác nhận trên MLflow, sau đó đối chiếu artifact. Xem kết quả cuối file.

## Cách dự báo đã chọn cho lần chạy đầu

Dùng **dự báo trực tiếp**, nghĩa là dự báo ngày thứ 1, 2, ..., 28 từ cùng
một mốc lịch sử. Thêm cột `horizon` để mô hình biết đang dự báo xa bao nhiêu
ngày. Một mô hình dùng chung cho 297 chuỗi và 28 bước; học riêng bốn mô hình
cho q05, q50, q90, q95 bằng `objective="quantile"`, `alpha` tương ứng.
Đây là lựa chọn thiết kế ban đầu, chưa chứng minh tốt hơn các cách khác.

Ví dụ dùng đúng mốc trong dự án: đứng ở cuối d_1605 để dự báo d_1606–d_1633.

- Với cả 28 ngày, chỉ dùng số bán và giá quan sát tới hết d_1605.
- Dự báo d_1606 có `horizon=1`; d_1633 có `horizon=28`.
- Thứ, tháng, sự kiện và SNAP là thông tin của **ngày cần dự báo**.
  Tiếp tục giả định lịch sự kiện/SNAP đã biết trước như tuần 3.
- Không đưa số bán thật d_1606 vào đầu vào khi dự báo d_1607.
  Không điền q95 hay q50 vào lịch sử để tính những ngày sau.

Cách này cho phép mô hình học phân vị cho từng bước dự báo từ lịch sử
quan sát tại mốc. Vẫn phải đo độ phủ; không bảo đảm các phân vị đã chuẩn.

Tham số thư viện xem [LightGBM — objective](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective)
và [alpha](https://lightgbm.readthedocs.io/en/stable/Parameters.html#alpha).

## Đầu vào mô hình

Có 23 đặc trưng, do `src/data/direct_features.py` tạo:

- `origin_sales`: số bán ngay tại ngày mốc.
- `origin_sales_lag_7/14/28`: số bán cách **ngày mốc** 7/14/28 ngày.
  Ví dụ mốc d_1605 thì lag 7 là d_1598, bất kể horizon bằng bao nhiêu.
- `origin_sales_roll_mean_7/28`: trung bình trong 7/28 ngày kết thúc ở
  **ngày mốc**, tính cả ngày mốc. Loại 25/12 khỏi phép trung bình theo
  giả định đóng cửa; lag và số bán tại mốc giữ nguyên số quan sát.
- `origin_sell_price`: giá quan sát tại ngày mốc, dùng cho cả cửa sổ.
  Không dùng giá thật sau mốc.
- Bốn mã sản phẩm/bộ phận/cửa hàng/bang; bốn cột sự kiện; bảy biến lịch
  và SNAP của ngày cần dự báo; thêm `horizon`.

Chỉ lấy origin sau khi đủ lịch sử kể từ lần bán dương đầu tiên. Danh sách
giá trị category chỉ học từ bảng train; category mới khi dự báo thành thiếu.
Các cột EDA `mean_live`, `volume_group`, `first_date` không làm đầu vào.

Bộ đặc trưng và cách huấn luyện khác tuần 3. Vì thế khi so sánh về sau,
không quy toàn bộ chênh lệch với LightGBM cũ cho riêng hàm mất mát phân vị.
Muốn tách riêng tác động của loss phải có thí nghiệm đối chứng phù hợp.

## Mốc huấn luyện và cấu hình xuất phát

Trong train, mỗi ví dụ có dạng:

```text
(lịch sử tại origin, lịch của origin+horizon, horizon)
    -> số bán thật tại origin+horizon
```

Chỉ dùng ví dụ có đủ cả khối nhãn 28 ngày kết thúc không muộn hơn d_1605.
Mốc cuối là d_1577; nhãn của khối cuối là d_1578–d_1605. Từ đó lùi mỗi
28 ngày để lấy mốc trước. Các mốc được chọn là d_37, d_65, ..., d_1577;
mỗi chuỗi chỉ tham gia khi đủ lịch sử. Không phải mọi chuỗi có mặt ở mốc sớm.

Khoảng cách mốc 28 ngày là lựa chọn giảm khối lượng tính toán. Các khối
nhãn không chồng lấn, nhưng lịch sử vẫn chồng lấn; mẫu không độc lập.
Chưa thử các khoảng cách khác hoặc tối ưu lựa chọn này.

`configs/quantiles.toml` lưu bốn mức phân vị, khoảng cách mốc và tham số:
500 cây, learning rate 0,05, tối đa 31 lá, min child samples 50, seed 29,
bốn luồng CPU. Đây đều là **lựa chọn kỹ thuật xuất phát**, không phải
tham số vận hành. 500 cây lấy làm ngân sách ban đầu từ tuần 3, chưa được
chọn lại cho pinball loss. Lần chạy này chưa dừng sớm hoặc chọn tham số.

## Chạy lần đầu

Tại thư mục gốc repo, trong terminal đã kích hoạt `.venv`:

```powershell
python -m src.forecast.quantile_model
```

Nếu chưa kích hoạt môi trường, dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m src.forecast.quantile_model
```

Chỉ cần chạy một trong hai lệnh. Lệnh sẽ học bốn mô hình, tạo dự báo cho
cửa sổ validation đầu d_1606–d_1633 và lưu một run MLflow mới. Chưa chấm
pinball loss/coverage, chưa dùng test. Chưa đo thời gian huấn luyện M5.

Tiêu chí kỹ thuật cần kiểm tra khi chạy:

- Có log `Training q05`, `q50`, `q90`, `q95` và số vòng thực tế sau mỗi mô hình.
- `Feature count: 23`; `Status: FINISHED`.
- `Forecast shape: (8316, 13)` = 297 chuỗi × 28 ngày, 13 cột.
- `Forecast series: 297`; `Forecast days: 1606 1633`.
- `Validation scored: false; test evaluated: false`.

8.316 là kích thước suy ra từ thiết kế, không phải số đo độ chính xác.
Run ID do MLflow tạo, mỗi lần chạy khác nhau. Nếu chạy lỗi, gửi traceback
để xử lý trước; không tự sửa theo nhãn validation hoặc thử test.

Trợ lý đã chạy riêng bước chuẩn bị bảng trên M5, **chưa fit mô hình**:
387.856 dòng train, 297 chuỗi, 56 mốc d_37–d_1577; nhãn d_38–d_1605.
Mỗi cặp chuỗi/mốc có đủ 28 nhãn, không thiếu đầu vào train. Bảng dự báo
đầu vào có đúng 8.316 dòng, 297 chuỗi, ngày d_1606–d_1633 và 23 đặc trưng.
Đây là kiểm tra kích thước dữ liệu; chưa phải pinball loss hoặc coverage.

Kiểm tra code: 114 tests toàn repo đạt; Ruff đạt trên các file mới. Tests
mới kiểm tra số tính tay, rò rỉ theo mốc, lịch sử riêng từng chuỗi, category
mới, và lưu/nạp bốn mô hình trên dữ liệu giả. Còn cảnh báo deprecation
SQLAlchemy bên trong MLflow như trước, không làm test thất bại.

## Kết quả được lưu ở đâu?

- Experiment: `week4-quantile-forecasting`, dùng chung cơ sở dữ liệu cục bộ
  `mlruns/mlflow.db`, không đổi experiment tuần 3.
- Trong artifacts: `models/q05.txt`, `q50.txt`, `q90.txt`, `q95.txt`;
  category levels, schema đặc trưng; cấu hình hiệu lực, phiên bản thư viện
  và bản chụp các file code/cấu hình.
- Dự báo CSV:
  `reports/quantiles/<run-id>/forecasts_valid_first_window.csv`.

CSV có năm cột khóa/thời gian (`id`, `origin_day`, `day`, `date`, `horizon`),
bốn cột `q05_raw`/`q50_raw`/`q90_raw`/`q95_raw` và bốn cột `q05`/`q50`/
`q90`/`q95` đã đặt giá trị âm về 0. Giữ số thực; không sắp xếp lại các
phân vị, để bước sau vẫn phát hiện được đảo thứ tự. Không có nhãn thật
validation trong file dự báo.

Mô hình là artifact text LightGBM. Khi nạp lại trên Windows, đọc bằng
`Path(path).read_text(encoding="utf-8")` rồi truyền `model_str` vào Booster
để chuẩn hóa xuống dòng, tương tự cách xử lý đã có ở tuần 3.

Cơ sở dữ liệu/mô hình trong `mlruns/` vẫn chỉ ở máy, được Git bỏ qua.
Commit code không thay thế việc sao lưu thư mục này.

## Điểm dừng

Gửi output terminal từ phần số dòng train đến phần Run ID/trạng thái và
đường dẫn dự báo. Có thể mở MLflow bằng lệnh đã dùng tuần 3, chọn experiment
`week4-quantile-forecasting` rồi kiểm tra bốn file mô hình.

Mục 2 hoàn tất khi lần chạy thật được xác nhận đủ bốn mô hình, schema và
dự báo. Sau đó mới sang mục 3: pinball loss, coverage, độ rộng và crossing.
Chưa kết luận độ tin cậy của khoảng chỉ từ việc train thành công.

## Kết quả đã xác nhận của mục 2

Run `584609c2de6a466ba50a3c3c25329566`, experiment
`week4-quantile-forecasting`, trạng thái `FINISHED`.

- Train: 387.856 dòng, 56 mốc, 23 đặc trưng, nhãn kết thúc d_1605.
- Đủ q05/q50/q90/q95; mỗi mô hình đã học 500 vòng.
- CSV: 8.316 dòng × 13 cột, 297 chuỗi, d_1606–d_1633, không trùng khóa
  `id + day`. File trên máy trùng bản lưu trong artifact MLflow.
- Người thực hiện đã thấy run trên giao diện MLflow dùng SQLite cục bộ.
- Chưa chấm validation hoặc dùng test trong run huấn luyện này.

Mục 2 hoàn tất. Mục 3 tiếp theo sẽ chấm pinball loss, độ phủ, độ rộng và
đảo thứ tự phân vị của dự báo đã lưu.
