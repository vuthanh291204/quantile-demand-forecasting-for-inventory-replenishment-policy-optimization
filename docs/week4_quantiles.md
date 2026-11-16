# Tuần 4 — Dự báo phân vị, pinball loss và độ phủ

Ngày bắt đầu hướng dẫn: 16/11/2026.

Trạng thái: mục 1 hoàn thành sau khi người thực hiện trả lời đúng bài tập.
Mục 2 đã được xác nhận bằng lần chạy M5 và giao diện MLflow.
[Mục 3A: chấm cửa sổ validation đầu](week4_evaluation.md) đã hoàn tất với
kết quả người thực hiện gửi và CSV/MLflow đã đối chiếu. Mục 3B đã hoàn
thành trên sáu cửa sổ validation, run `6677b9628294457fb1a3df04056875d1`.
Kết quả được ghi ở `docs/week4_evaluation.md`.

Mục 4 đã xuất đủ bốn hình ở run `acc89fc3c0174e42ab547c44f0bd7c9b`.
Mục 3C đã hoàn tất trên test ở run `6bb24de09d644f58bbaa63ab0a608ad2`.
Mục lớn 3 và 4 đã hoàn tất. Tiếp tục mục 5 để nhận xét độ phủ khoảng 90%.
Các số trong ví dụ và bài tập dưới đây đều là **số tự đặt để minh họa**.

## 1. Những gì có thể dùng tiếp từ tuần 1–3

- Tập con cố định: 297 chuỗi FOODS ở CA_3, TX_2, WI_1; mỗi cửa hàng có
  33 chuỗi ở mỗi mức bán thấp/vừa/cao. Seed chọn mẫu là 29.
- Hai notebook đã chọn mẫu, nối doanh số với lịch và giá, rồi khám phá dữ liệu.
  `data/processed/subset_long.parquet` có 576.477 dòng, gồm 297 × 1.941 ngày.
- Train d_1–d_1605; validation d_1606–d_1773; test d_1774–d_1941.
  Validation và test đều có sáu cửa sổ 28 ngày, theo `configs/split.toml`.
- `src/data/features.py` tạo 21 đặc trưng, giới hạn lịch sử trước mốc dự báo,
  xóa doanh số tương lai và giữ giá ở mức cuối cùng đã biết.
- `src/forecast/lightgbm_model.py` có chuẩn bị dữ liệu, danh sách category
  chỉ học từ lịch sử và vòng dự báo điểm đệ quy 28 ngày.
- Có thước đo dự báo điểm, kiểm tra rò rỉ, kiểm tra khóa chuỗi/ngày và MLflow.
  `src/evaluation/metrics.py` hiện chưa có pinball loss hoặc coverage.
- CSV so sánh validation xác nhận RMSSE trung bình: ETS 0,673912;
  trung bình trượt 28 ngày 0,677792; LightGBM 500 cây 0,694558.
  LightGBM đứng thứ ba theo thước đo này, chưa chứng minh thắng baseline.
- Baseline đã được đánh giá trên test ở tuần 2. LightGBM mới đánh giá
  trên validation. Không mô tả toàn bộ tập test là chưa từng được xem.

Nguồn đối chiếu: `reports/week2_baselines.md`, `reports/week3_lightgbm.md`,
`reports/baselines/run_info.json` và
`reports/lightgbm/08e87e3b435c4aa6a8cb5809d2ab07fb/comparison_valid.csv`.

Các giới hạn đã có vẫn áp dụng: doanh số được coi là nhu cầu; 25/12 được
giả định đóng cửa; lịch sự kiện/SNAP được giả định biết trước. Chọn tập con
và mức bán ban đầu đã nhìn toàn giai đoạn. Các mức bán đó không được đưa
vào làm đặc trưng mô hình. Chưa có kết quả chi phí hay mức đáp ứng từ mô phỏng.

## 2. Phân vị: một ngưỡng gắn với một mức xác suất

Dự báo điểm trả lời: “Ngày mai dự kiến bán bao nhiêu?”. Dự báo phân vị
bổ sung câu hỏi: “Ngưỡng nào ứng với một mức xác suất không vượt quá?”.

Ký hiệu `q90` là dự báo phân vị 90%, tức `tau = 0.90`. Ví dụ minh họa,
`q90 = 8` đặt mục tiêu rằng số bán không vượt 8 trong khoảng 90% các
trường hợp có thông tin đầu vào tương ứng. Đây là mục tiêu mô hình cần học;
phải đo trên dữ liệu ngoài train để biết nó có làm được không.

- `q50`: trung vị có điều kiện; khác với trung bình mà LightGBM L2 tuần 3
  nhắm tới. Không đổi tên dự báo L2 cũ thành q50.
- `q90`, `q95`: các ngưỡng trên để mô tả phía nhu cầu cao.
- `q05`: ngưỡng phía nhu cầu thấp, có thể dùng làm đầu dưới của khoảng.

Với số bán nguyên và nhiều ngày bằng 0, xác suất không vượt một ngưỡng có
thể nhảy qua mức mong muốn. Ví dụ **phân phối giả định**: bán 0 với xác suất
80%, bán 1 với xác suất 20%. Trung vị là 0 nhưng xác suất bán không quá 0
là 80%. Vì thế không bắt buộc tỷ lệ `sales <= q50` đúng 50% khi có nhiều
giá trị trùng nhau. Với phân vị đúng, mức 'tau' nằm giữa xác suất nhỏ hơn
ngưỡng và xác suất nhỏ hơn hoặc bằng ngưỡng.

## 3. Phân biệt q90 và khoảng dự báo 90%

`q90` là một đầu mút. Khoảng dự báo có hai đầu mút.

Cách xây dựng khoảng trung tâm với mức phủ danh nghĩa 90% là:

```text
đầu dưới = q05
đầu trên = q95
khoảng = [q05, q95]
mức phủ danh nghĩa = 0.95 - 0.05 = 0.90
```

“Danh nghĩa” nghĩa là mức đặt mục tiêu trước khi đo. Độ phủ thật có thể
khác vì mô hình sai, dữ liệu hữu hạn, thay đổi theo thời gian và tính rời rạc.
Trong ví dụ phân phối 0/1 ở trên, [q05, q95] = [0, 1] phủ 100% dù các
phân vị đều đúng. Không tự kết luận mô hình sai chỉ vì độ phủ vượt 90%.

`[q50, q95]` có chênh lệch mức phân vị 45 điểm phần trăm, không phải cách
xây dựng khoảng trung tâm 90%. `[0, q90]` là khoảng một phía khi nhu cầu
không âm; nếu dùng phải ghi rõ đang đánh giá ngưỡng trên.

**Lựa chọn xuất phát cho mục 2, chưa tối ưu:** học q05, q50, q90, q95. Thêm q05
để vẽ khoảng trung tâm 90%, đồng thời giữ các mức ví dụ trong file tiến độ.
Các mức này là lựa chọn thí nghiệm, không phải mức đáp ứng đơn đã đạt.

Cách ghép hai phân vị và cách đo độ phủ được minh họa trong
[tài liệu chính thức scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).

## 4. Pinball loss: phạt thiếu và thừa khác nhau

Gọi `y` là số bán thật, `q` là dự báo tại mức `tau`:

```text
nếu y >= q: loss = tau × (y - q)
nếu y <  q: loss = (1 - tau) × (q - y)
```

Ở q90, dự báo thấp hơn thực tế bị phạt với hệ số 0,9, cao hơn thực tế
bị phạt với hệ số 0,1. Cùng độ lệch, trường hợp dự báo thiếu bị phạt nặng
hơn để mô hình học một ngưỡng phía trên.

Ví dụ tính tay, **không phải kết quả M5**:

- `tau = 0.9`, `y = 8`, `q = 5`: thiếu 3, loss = 0,9 × 3 = 2,7.
- `tau = 0.9`, `y = 2`, `q = 5`: thừa 3, loss = 0,1 × 3 = 0,3.
- Trung bình hai trường hợp: (2,7 + 0,3) / 2 = 1,5.

Loss càng thấp càng tốt khi so các mô hình ở **cùng mức phân vị, dữ liệu
và cách gộp**. Pinball loss ở q50 bằng một nửa MAE. Loss không phải phần
trăm chính xác hoặc chi phí tồn kho. Các hệ số 0,9/0,1 là trọng số của
hàm học, chưa phải giả định chi phí thiếu hàng/lưu kho.

Tham khảo công thức:
[scikit-learn — Pinball loss](https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss).

## 5. Coverage: đếm tỷ lệ giá trị thật nằm trong khoảng

Với khoảng `[lower, upper]`, một quan sát được phủ khi:

```text
lower <= y <= upper
coverage = số quan sát được phủ / số quan sát được đánh giá
```

Tính cả trường hợp bằng đầu mút; quy tắc này đặc biệt đáng chú ý khi số
bán là số nguyên. Ví dụ minh họa: trong 20 ngày được chấm có 18 ngày nằm
trong khoảng thì coverage = 18/20 = 90%.

Đo thêm độ rộng: `width = upper - lower`, tính theo đơn vị hàng. Một khoảng
rất rộng có thể phủ nhiều nhưng ít hữu ích. Cần xem đồng thời pinball loss,
độ phủ, độ rộng và tỷ lệ nằm dưới/trên khoảng.

Đối với một phân vị riêng, dùng tên rõ ràng như
`upper_coverage_q90 = mean(y <= q90)` để tránh nhầm với độ phủ khoảng
`interval_coverage_90 = mean(q05 <= y <= q95)`. Khi đánh giá phân vị với
số bán rời rạc, xem thêm `mean(y < q90)` để nhận diện ảnh hưởng của giá
trị trùng đầu mút. Cả hai đều là tỷ lệ số lần, không phải tỷ lệ lượng hàng.

Độ phủ gộp có thể che khuất chuỗi bán ít hoặc những ngày dự báo xa bị kém.
Ở bước đánh giá cần lưu số ngày được chấm, kết quả từng chuỗi/cửa sổ và
theo bước dự báo 1–28; ghi rõ cách gộp. Giữ quy tắc loại 25/12 như tuần 2–3.

## 6. Những chỗ phải cẩn thận khi sang mục huấn luyện

- LightGBM hỗ trợ `objective="quantile"` và `alpha` là mức phân vị. Mỗi
  mức được học bằng một mô hình trong phương án ban đầu. Chưa có mô hình
  phân vị nào được huấn luyện ở bước học khái niệm này.
- 500 cây đã chọn cho L2 không tự động tối ưu cho pinball loss. Nếu dùng
  làm cấu hình xuất phát thì ghi rõ là lựa chọn ban đầu; chỉ dùng validation
  để lựa chọn tiếp, không chọn bằng RMSSE của q90/q95 hoặc bằng điểm test.
- Vòng `recursive_forecast()` hiện điền dự báo điểm vào lịch sử tương lai.
  Thay mô hình bằng q95 rồi liên tục điền q95 không tự tạo ra phân vị 95%
  đúng cho ngày thứ 28. Cần thiết kế rõ đầu vào theo mốc và cách dự báo
  nhiều ngày ở mục 2; không mặc định đổi objective là hoàn tất.
- Các mô hình học riêng có thể cho q05 > q50 hoặc q90 > q95, gọi là
  **đảo thứ tự phân vị** (quantile crossing). Cần đo tỷ lệ này; nếu có
  hậu xử lý, lưu cả kết quả gốc và mô tả quy tắc đã dùng.
- Dự báo phân vị theo từng ngày chưa phải phân vị tổng nhu cầu trong thời
  gian chờ hàng. Không mặc định cộng q95 của bảy ngày thành q95 của tổng.
- q95 hoặc coverage 95% không bảo đảm đáp ứng được 95% lượng hàng khách
  cần. Mức đáp ứng phải được định nghĩa và đo trong mô phỏng kho sau này.

Tham số mô hình tham khảo:
[LightGBM — objective](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective)
và [alpha](https://lightgbm.readthedocs.io/en/stable/Parameters.html#alpha).
Các lưu ý về vòng đệ quy là nhận xét từ luồng code hiện tại, chưa phải
kết luận từ thí nghiệm tuần 4.

## 7. Thứ tự làm tuần 4 và điểm dừng

1. Học phân vị, pinball loss và coverage: đọc tài liệu này, gửi bài tập
   ở mục 8. Chỉ xác nhận hoàn tất sau khi kiểm tra cách hiểu.
2. Huấn luyện LightGBM phân vị: thiết kế dự báo nhiều ngày, tạo cấu hình
   riêng, kiểm tra rò rỉ, chạy một cửa sổ validation trước; gửi log và Run ID.
3. Tính pinball loss, coverage, độ rộng và crossing trên validation.
   Đánh giá đủ sáu cửa sổ sau khi kiểm tra lần chạy nhỏ. Chỉ đánh giá test
   sau khi khóa cấu hình và quy tắc dự báo; kết quả test không dùng để chỉnh lại.
4. Vẽ số bán thật với q50 và [q05, q95] cho một số chuỗi được chọn rõ lý do.
5. Nhận xét độ phủ danh nghĩa 90% so với độ phủ đo được, độ rộng, tính rời
   rạc, mức bán và bước dự báo. Không tự bịa một ngưỡng sai lệch “đạt chuẩn”.
6. Chuẩn bị bản tóm tắt Mốc 1 từ kết quả đã xác nhận. Người thực hiện gửi
   thầy cô và xin góp ý sau khi đọc bản tóm tắt.

Mỗi đầu mục dừng để người thực hiện gửi kết quả trước khi sang đầu mục sau.
Chủ động commit bằng tiếng Anh sau khi hoàn tất mục lớn, theo yêu cầu
được người thực hiện cập nhật ngày 16/11/2026. Với mục 3, mốc commit là
sau khi hoàn thành cả phần validation và test. Mục học khái niệm đã hoàn tất.

## 8. Bài tập gửi lại trước khi sang mục 2

Các số dưới đây là **giả định minh họa**, không lấy từ dữ liệu hoặc mô hình.
Có thể tính tay, chưa cần chạy Python.

1. Mô hình cho `q50 = 3`, `q90 = 7`, `q95 = 9`. Có thể gọi `[3, 9]` là
   khoảng trung tâm 90% không? Muốn tạo khoảng trung tâm 90%, còn thiếu
   phân vị nào?
2. Với `tau = 0.9`, tính pinball loss của hai trường hợp: `y = 10, q = 6`
   và `y = 2, q = 6`. Trường hợp nào bị phạt nặng hơn, vì sao?
3. Trong 20 ngày được chấm, 17 ngày nằm trong `[q05, q95]`. Coverage bằng
   bao nhiêu? Có thể suy ra kho đáp ứng được cùng tỷ lệ lượng hàng khách
   cần không?

Kết quả người thực hiện đã gửi và được xác nhận đúng:

1. Còn thiếu q05. Bổ sung rõ: [q50, q95] không phải khoảng trung tâm 90%;
   cần dùng [q05, q95].
2. Hai loss lần lượt là 3,6 và 0,4; trường hợp dự báo thiếu bị phạt nặng hơn.
3. Coverage = 85%; không suy ra được tỷ lệ lượng hàng được đáp ứng.

Mục 1 hoàn tất. Người thực hiện đã yêu cầu chuyển sang mục 2.
