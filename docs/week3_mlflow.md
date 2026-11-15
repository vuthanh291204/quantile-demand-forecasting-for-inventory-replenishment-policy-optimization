# Tuần 3 — Mục 3: ghi thí nghiệm bằng MLflow

Mục tiêu của bước này: lưu được một lần chạy, đọc lại đúng thông tin đã lưu
và mở được giao diện xem. Khi làm mục 4–5, cùng cơ chế này sẽ ghi cấu hình
LightGBM và điểm dự báo đo trên validation.

## Năm khái niệm cần biết

- **Experiment:** nhóm các lần chạy cùng bài toán. Ở đây đặt tên
  `week3-demand-forecasting`; tên này là lựa chọn tổ chức dự án.
- **Run:** một lần chạy cụ thể, có mã `Run ID` riêng. Chạy lại cùng cấu hình
  vẫn tạo một run mới để giữ lịch sử.
- **Params:** các lựa chọn đầu vào, ví dụ lag 7/14/28, độ dài cửa sổ dự báo.
  Khi có LightGBM sẽ thêm các tham số như số lá và tốc độ học.
- **Metrics:** các con số đo được từ lần chạy. Hiện ghi số dòng, số chuỗi
  và số cửa sổ đã kiểm tra; sau này mới có MAE/RMSSE đo từ mô hình.
- **Artifacts:** các file đi kèm lần chạy, ví dụ cấu hình, báo cáo, mã nguồn
  và sau này là file mô hình hoặc dự báo.

MLflow giúp trả lời: lần chạy nào dùng cấu hình gì và cho kết quả nào.
Nó không tự ngăn rò rỉ hay bảo đảm mô hình đúng; các kiểm tra ở mục 2 vẫn cần.

## Cách lưu đã chọn

MLflow đã có trong `.venv`. Cấu hình ở `configs/tracking.toml`:

- `mlruns/mlflow.db`: SQLite lưu thông tin run, params, metrics và tags.
- `mlruns/artifacts/`: lưu các file đính kèm.
- `src/tracking.py`: hàm `configure_tracking()` để tái sử dụng khi huấn luyện,
  cùng lệnh kiểm tra việc ghi dữ liệu.

Đây là lựa chọn chạy cục bộ cho một người. Không cần đăng ký tài khoản hoặc
cài máy chủ cơ sở dữ liệu riêng. Cách dùng SQLite và mở giao diện được đối
chiếu với [tài liệu chính thức MLflow](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/).
Đường dẫn của script được tính từ thư mục repo để tránh ghi nhầm cơ sở dữ liệu.

`mlruns/` đã có trong `.gitignore`. Commit lưu code và cấu hình;
lịch sử các run vẫn nằm trong thư mục này trên máy, không đi kèm commit.

## Bước 1: ghi một lần chạy kiểm tra

Tại thư mục gốc repo, trong terminal đã kích hoạt `.venv`:

```powershell
python -m src.tracking
```

Lệnh đọc báo cáo `reports/leakage_checks.json` đã có từ mục 2. Nó kiểm tra
báo cáo đủ sáu cửa sổ, các kiểm tra đều đạt và cấu hình khớp hiện tại,
rồi ghi một run tên `tracking-preflight`.

Các con số dưới đây lấy từ báo cáo thật đã lưu, không phải điểm dự báo giả:

```text
Experiment: week3-demand-forecasting
Run ID: <mã riêng cho lần chạy của bạn>
Status: FINISHED
audit_n_features: 21
audit_n_series: 297
audit_synthetic_rows_checked: 49896
audit_train_rows: 391610
audit_validation_windows: 6
```

`audit_synthetic_rows_checked` là số dòng dùng dãy giả để kiểm tra tính đặc
trưng ở mục 2. Nó không phải độ chính xác của mô hình. Run hiện tại cũng chưa
có mô hình huấn luyện, nên không ghi MAE/RMSSE.

File đính kèm gồm báo cáo kiểm tra; cấu hình tracking, đặc trưng, cách chia;
danh sách chuỗi; mã nguồn liên quan; phiên bản Python/thư viện và mã băm
SHA-256 của các file chụp lại. Tags có Git commit và cờ `git_dirty`:
`true` nghĩa là lúc chạy có thay đổi chưa commit, nên cần xem cả bản mã nguồn
đính kèm nếu muốn tái tạo chính xác.

Lệnh này không chạy lại kiểm tra rò rỉ. Nếu đã sửa dữ liệu hoặc code đặc trưng,
chạy lại `python -m src.evaluation.check_leakage` trước khi ghi báo cáo mới.
So khớp cấu hình không đủ để chứng minh báo cáo còn đúng sau mọi thay đổi.

## Bước 2: mở giao diện

Vẫn từ thư mục gốc repo và `.venv`:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Giữ terminal đang chạy, mở `http://127.0.0.1:5000` trong trình duyệt.
Lệnh này đọc cùng cơ sở dữ liệu mà script đã ghi. Không chỉ chạy `mlflow ui`
với đường dẫn mặc định, vì có thể mở nhầm kho lưu trữ và không thấy run.

Trong giao diện:

1. Chọn experiment `week3-demand-forecasting` ở phần Experiments.
2. Mở run `tracking-preflight` mới nhất, đối chiếu `Run ID` với terminal.
3. Xem Params: `lags`, `rolling_windows`, `horizon_days`, `price_policy`.
4. Xem Metrics: năm số có tiền tố `audit_` như ở trên.
5. Xem Artifacts: mở thư mục `configs`, `code`, `audit` và file `environment.json`.

Đã có một run kiểm tra do trợ lý chạy để xác nhận lưu/đọc thành công.
Sau khi bạn chạy, có thêm run mới là bình thường. Lần đầu có thể thấy log tạo
bảng SQLite; tiêu chí thành công của script là `Status: FINISHED` và các số
được đọc lại đúng từ MLflow.

Dừng giao diện bằng Ctrl+C trong terminal. Các run đã lưu vẫn còn để mở lại.
Nếu cổng 5000 đã được dùng, đổi `--port 5001` rồi mở `http://127.0.0.1:5001`.

## Điểm dừng và mốc commit

Gửi output của `python -m src.tracking` cùng xác nhận bạn đã thấy run và các
file trong giao diện. Mục 3 hoàn tất khi cả ghi/đọc dữ liệu và mở giao diện
được xác nhận. Sau đó commit code/cấu hình mục này bằng tiếng Anh, ví dụ:

```text
Set up local MLflow experiment tracking
```

Mốc trước đã commit: `0159df7` — `Add leakage-safe forecasting features and validation audit`.
Người thực hiện đã chạy thành công, nhận `Status: FINISHED` cho run
`a569b14df9d649918fda5e430c9f2eba` và xác nhận thấy run trong giao diện.
Mục 3 hoàn thành. Mục 4 sẽ được hướng dẫn từng bước để người thực hiện tự viết
và chạy code; chỉ làm thay khi được yêu cầu.

Mục 3 đã commit trong `44e6fd6` — `Set up local MLflow experiment tracking`.
Tiến độ LightGBM được ghi tiếp ở [week3_lightgbm.md](week3_lightgbm.md).
