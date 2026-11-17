# Tuần 5 — Bộ mô phỏng tồn kho

Mục 1 đã được đối chiếu với bài tập người thực hiện gửi. Lượng thiếu ở
câu 1 được sửa từ -3 thành 0 và giải thích bên dưới. Mục 2 hoàn tất sau
khi người thực hiện chọn lost sales. Code mục 3 đã vượt qua kiểm tra,
đang chờ output chạy ví dụ của người thực hiện để xác nhận đầu mục.
Các đầu mục được thực hiện lần lượt theo yêu cầu của người thực hiện.

Theo yêu cầu cập nhật ngày 17/11/2026, khi hoàn tất một đầu mục có code,
kiểm tra phần triển khai rồi commit bằng tiếng Anh.

## Điểm bắt đầu đã đối chiếu

- Hai notebook tạo tập con 297 chuỗi FOODS ở CA_3, TX_2, WI_1. Một chuỗi
  tương ứng một cặp sản phẩm–cửa hàng. `subset_long.parquet` có 576.477
  dòng, đủ 1.941 ngày cho mỗi chuỗi, không trùng khóa chuỗi/ngày.
- Cấu hình chia thời gian giữ train đến d_1605, validation d_1606–d_1773,
  test d_1774–d_1941. Mỗi tập đánh giá có sáu cửa sổ 28 ngày.
- Baseline, LightGBM điểm, LightGBM phân vị và báo cáo tuần 1–4 đã có.
  Dự báo phân vị hiện được cập nhật ở đầu mỗi khối 28 ngày.
- Khi bắt đầu tuần 5, `src/simulation/__init__.py` rỗng. Mục 3 bổ sung
  `src/simulation/inventory.py` để mô phỏng một sản phẩm với lịch đặt sẵn.
- Kiểm tra hiện trạng bằng Python trong `.venv` cho kết quả 150 tests đạt.
  Có một cảnh báo deprecation từ SQLAlchemy bên trong MLflow.

Nguồn trong dự án gồm hai notebook, hai parquet tập con,
`configs/subset_ids.csv`, `configs/split.toml`, mã trong `src/`,
[giả định đã ghi](assumptions.md) và [bản tóm tắt Mốc 1](../reports/milestone1_summary.md).

## Các đầu mục tuần 5

- [x] 1. Học vòng nhận hàng, bán, ghi nhận thiếu và xét đặt hàng hằng ngày,
  đối chiếu bài tập và giải thích điều chỉnh lượng thiếu ở câu 1.
- [x] 2. Chọn lost sales và ghi giả định ở mục 28 của `docs/assumptions.md`.
- [ ] 3. Hàm mô phỏng đã triển khai và kiểm tra, chờ người thực hiện gửi
  output ví dụ để xác nhận đầu mục.
- [ ] 4. Thiết kế giao diện chính sách đặt hàng.
- [ ] 5. Viết test với đáp án tính tay.
- [ ] 6. Chạy chính sách đơn giản và vẽ tồn kho theo ngày.
- [ ] 7. Viết đoạn báo cáo mô tả mô phỏng và giả định.

## Mục 1 — Theo dõi một sản phẩm qua từng ngày

Hãy hình dung một mặt hàng tại một cửa hàng. Dữ liệu cho biết số lượng
bán theo ngày. Theo **giả định đã có của dự án**, ta dùng số lượng này
làm nhu cầu đầu vào để thử các cách đặt hàng.

Cần theo dõi ba thông tin trước khi xử lý ngày tiếp theo.

- **Hàng có sẵn** (`on_hand`) là số đơn vị đang có thể bán ngay.
- **Hàng đang về** (`on_order`, còn gọi là pipeline) gồm những đơn đã
  đặt nhà cung cấp. Mỗi đơn cần có số lượng và ngày dự kiến nhận.
- **Nhu cầu chưa đáp ứng** (`unmet_demand`) là số đơn vị khách muốn mua
  nhưng kho chưa giao được trong ngày. Cách xử lý phần này được chọn ở mục 2.

Các khái niệm hàng có sẵn, hàng đang về và nợ đơn được trình bày trong
giáo trình của Manuel D. Rossetti. [Analysis of Inventory Systems, mục 1.3](https://rossetti.github.io/RossettiInventoryBook/01-Chapter1.html)

### Thứ tự xử lý một ngày

Thứ tự sau là **quy ước mô phỏng đề xuất**, phù hợp với đầu mục tuần 5
trong kế hoạch. M5 cung cấp doanh số tổng hợp theo ngày, nên thứ tự này
cần được ghi thành giả định khi chốt thiết kế.

1. **Đầu ngày nhận hàng.** Lấy các đơn đến hạn, cộng số lượng vào hàng
   có sẵn rồi xóa các đơn đã nhận khỏi danh sách hàng đang về.
2. **Đáp ứng nhu cầu trong ngày.** Trong ví dụ chỉ có nhu cầu hôm nay,
   lượng đáp ứng bằng số nhỏ hơn giữa nhu cầu và hàng có sẵn sau nhận hàng.
3. **Ghi nhận kết quả.** Trừ lượng đã giao khỏi hàng có sẵn, ghi lượng
   thiếu bằng nhu cầu trừ lượng đã đáp ứng. Nếu chọn nợ đơn ở mục 2,
   vòng lặp sẽ cần xử lý thêm phần nợ từ ngày trước.
4. **Cuối ngày xét đặt hàng.** Chính sách dùng trạng thái kho và thông tin
   đã biết đến cuối ngày để quyết định lượng đặt. Đơn mới được thêm vào
   danh sách hàng đang về với ngày nhận cụ thể.

Simulator sẽ đọc nhu cầu từng ngày để tái hiện việc bán hàng. Đầu vào
của chính sách tại cuối ngày t chỉ gồm thông tin đã có đến ngày t.

### Ví dụ tính tay

**Tất cả số sau là số tự đặt để học, chưa phải tham số vận hành đã chốt.**

Ở ngày 10, kho có sẵn 4 đơn vị, nhận thêm 3 đơn vị vào đầu ngày và có
nhu cầu 5 đơn vị trong ngày.

- Sau nhận hàng có `4 + 3 = 7` đơn vị.
- Đáp ứng `min(7, 5) = 5` đơn vị.
- Thiếu `5 - 5 = 0` đơn vị, tồn cuối ngày `7 - 5 = 2` đơn vị.
- Giả sử cuối ngày quyết định đặt thêm 6 đơn vị, hàng có sẵn lúc đó vẫn
  là 2 đơn vị, đồng thời danh sách hàng đang về có thêm đơn 6 đơn vị.

Để ghi ngày nhận, **ví dụ dùng quy ước** đặt cuối ngày t với thời gian
chờ L thì nhận đầu ngày t + L, với L là số nguyên từ 1 trở lên. Nếu L = 2,
đơn đặt cuối ngày 10 sẽ nhận đầu ngày 12. L ở đây tính theo chênh lệch
chỉ số ngày. Quy ước này sẽ được chốt và kiểm tra bằng test khi triển khai.

### Khi xét đặt hàng cần nhìn cả hàng đang về

Một thông tin thường dùng là **vị thế tồn kho** (`inventory_position`).
Nếu phần thiếu được ghi thành nợ đơn phải giao sau, công thức là hàng
có sẵn cộng hàng đang về rồi trừ lượng nợ. Trong ví dụ có nợ bằng 0,
sau khi đặt đơn mới, vị thế bằng `2 + 6 = 8` đơn vị. Chính sách có thể
dựa vào thông tin này để tránh đặt lặp cho lượng đã được đặt trước.
[Analysis of Inventory Systems, mục 1.3](https://rossetti.github.io/RossettiInventoryBook/01-Chapter1.html)

### Đơn vị của mức đáp ứng trong đồ án

Với dữ liệu tổng hợp theo ngày, cách diễn đạt chính xác cho đồ án là
**tỷ lệ lượng hàng được đáp ứng ngay trên tổng lượng hàng được yêu cầu**
(`unit fill rate`). Ví dụ tự đặt, đáp ứng 95 trong tổng nhu cầu 100 đơn
vị hàng cho tỷ lệ 95%. Cách đếm theo lượng hàng được định nghĩa trong
tài liệu về fill rate của Paul A. Jensen. [Operations Research Models and Methods](https://utw11041.utweb.utexas.edu/ORMM/omie/operation/unit/inventory/deterministic/results_short.html)

## Bài tập gửi lại để kiểm tra mục 1

**Các số đều là giả định để tính tay.** Đầu ngày 1, kho có sẵn 5 đơn vị,
nhận thêm 2 đơn vị. Nhu cầu ngày 1 là 4 đơn vị. Có một đơn 6 đơn vị khác
sẽ về đầu ngày 3. Giả sử cuối ngày 1 đặt thêm 8 đơn vị, L = 2 theo quy
ước ngày nhận ở trên. Trong bài này lượng nợ đơn ban đầu bằng 0.

1. Sau khi bán ngày 1, đã đáp ứng bao nhiêu, thiếu bao nhiêu và tồn bao nhiêu?
2. Sau khi đặt đơn mới, tổng hàng đang về và vị thế tồn kho bằng bao nhiêu?
   Đơn mới đến đầu ngày nào?
3. Xét riêng một ngày khác có sẵn 3 đơn vị, không nhận hàng, nhu cầu 5
   đơn vị và không có nợ đầu ngày. Lượng đáp ứng, lượng thiếu và tồn
   cuối ngày lần lượt bằng bao nhiêu?

### Đối chiếu bài tập người thực hiện gửi

- Câu 1 trả lời đáp ứng 4, thiếu -3, tồn 3. Lượng đáp ứng và tồn đúng.
  Lượng thiếu cần sửa thành `nhu cầu - lượng đáp ứng = 4 - 4 = 0`.
  Có thể tính tương đương bằng `max(4 - (5 + 2), 0) = 0`.
  Phép trừ `4 - 7 = -3` cho biết hàng có sẵn dư 3 đơn vị so với nhu cầu.
- Câu 2 trả lời tổng hàng đang về 14 và đơn mới đến ngày 3, đều đúng.
  Vị thế tồn kho sau đặt hàng là `3 + 14 = 17` đơn vị khi nợ đơn bằng 0.
- Câu 3 trả lời đáp ứng 3, thiếu 2, tồn 0, đều đúng.

Lượng thiếu đếm số đơn vị chưa đáp ứng nên luôn từ 0 trở lên. Các đáp án
trên dùng các số giả định của bài tập.

## Mục 2 — Chọn cách xử lý lượng thiếu

Hai giả định dẫn đến cách cập nhật trạng thái khác nhau.

- **Mất lượt bán** (`lost sales`). Phần nhu cầu chưa đáp ứng được ghi
  nhận là lượng bán mất của ngày đó. Mỗi ngày sau xử lý nhu cầu mới của
  ngày đó.
- **Nợ đơn** (`backorders`). Phần chưa đáp ứng được giữ thành lượng còn
  nợ khách. Khi có hàng, mô phỏng cần quy tắc giao phần nợ và ghi lượng
  nợ còn lại.

Đây là hai cách mô hình hóa phản ứng khi thiếu hàng được trình bày trong
tài liệu của Paul A. Jensen. [Operations Research Models and Methods](https://utw11041.utweb.utexas.edu/ORMM/omie/computation/unit/invent_add/det_short.html)

**Đã được người thực hiện chọn ngày 17/11/2026**, dùng lost sales.
Dữ liệu M5 đang dùng gồm doanh số theo ngày, lịch và giá.
Chọn lost sales giúp theo dõi trực tiếp lượng yêu cầu và lượng đáp ứng
từng ngày với một quy tắc xử lý thiếu rõ ràng. Hành vi rời đi khi thiếu
hàng là **giả định mô phỏng tự chọn**.

Ví dụ tiếp từ câu 3, **vẫn là số giả định**, hôm nay thiếu 2 đơn vị,
ngày mai nhận 4 đơn vị và có nhu cầu mới 1 đơn vị.

- Với lost sales, ngày mai đáp ứng 1 đơn vị, tồn cuối ngày 3 đơn vị.
  Hai đơn vị bán mất hôm nay được giữ trong thống kê ngày hôm nay.
- Với backorders và **giả định ưu tiên trả nợ**, ngày mai giao 2 đơn vị
  nợ rồi đáp ứng 1 đơn vị nhu cầu mới, còn 1 đơn vị trong kho.

Giả định và lý do đã được ghi ở mục 28 của `docs/assumptions.md`.

## Mục 3 — Hàm mô phỏng một sản phẩm

Code ở `src/simulation/inventory.py`, với hàm `simulate_inventory()`.
Hàm nhận một chuỗi nhu cầu và lịch đặt hàng đã biết trước, rồi xử lý
nhận hàng, đáp ứng nhu cầu và đặt hàng theo từng ngày. Việc quyết định
lượng đặt từ trạng thái kho được bổ sung ở đầu mục chính sách tiếp theo.

Đầu vào gồm các thông tin sau.

- `demand` là số đơn vị yêu cầu từng ngày, từ ngày `start_day` trở đi.
- `initial_stock` là tồn trước khi nhận hàng ngày đầu.
- `lead_time_days` là L, nhận đầu ngày t + L khi đặt cuối ngày t.
- `order_quantities` là lượng đặt cuối từng ngày. Khi bỏ qua, toàn bộ
  lượng đặt bằng 0.
- `initial_orders` là ánh xạ ngày đến sang tổng lượng đã đặt từ trước.
  Ví dụ `{1: 2, 3: 6}` nghĩa là nhận 2 đơn vị ngày 1 và 6 đơn vị ngày 3.
- `start_day` mặc định là 1. Có thể truyền chỉ số ngày M5 như 1606 khi
  chuẩn bị mô phỏng trên validation.

Số lượng đầu vào phải là số nguyên không âm. L và chỉ số ngày phải là
số nguyên dương. Lịch đặt phải dài bằng chuỗi nhu cầu. Đơn có sẵn phải
có ngày nhận từ ngày bắt đầu trở đi. Hàm kiểm tra các điều kiện này
trước khi chạy và giữ nguyên dữ liệu người gọi truyền vào.

Kết quả có hai phần.

- `daily` là bảng theo ngày, gồm tồn đầu ngày (`opening_stock`), lượng
  nhận (`received`), hàng có thể bán (`available_stock`), nhu cầu
  (`demand`), đáp ứng (`fulfilled`), bán mất (`lost_sales`), tồn cuối
  (`closing_stock`), lượng đặt (`order_placed`), hàng đang về (`on_order`)
  và vị thế tồn kho (`inventory_position`). Hai cột cuối được ghi sau
  khi đặt hàng cuối ngày.
- `pending_orders` giữ ngày đến và lượng của các đơn còn đang về sau
  ngày mô phỏng cuối cùng.

Các đẳng thức dùng để kiểm tra là `demand = fulfilled + lost_sales`,
`closing_stock = opening_stock + received - fulfilled` và
`inventory_position = closing_stock + on_order`.

Tại thư mục gốc repo, với môi trường `.venv` đã kích hoạt, chạy ví dụ sau.

```powershell
python -m src.simulation.inventory
```

**Toàn bộ đầu vào của ví dụ là giả định minh họa**. Nhu cầu 5 ngày là
`[4, 5, 1, 8, 2]`, tồn ban đầu 5, L = 2, đơn có sẵn `{1: 2, 3: 6}` và
lượng đặt cuối ngày `[8, 0, 0, 4, 0]`.

Đáp án tính tay dự kiến như sau.

- Tồn cuối 5 ngày lần lượt là `3, 0, 13, 5, 3`.
- Lượng bán mất lần lượt là `0, 2, 0, 0, 0`.
- Ngày 3 nhận tổng cộng 14 đơn vị từ đơn có sẵn và đơn đặt ngày 1.
- Tổng lượng đáp ứng là 18, lượng bán mất là 2.
- Cuối kỳ còn đơn `{6: 4}`, tương ứng đơn đặt cuối ngày 4.

Người thực hiện gửi lại output lệnh trên để đối chiếu trước khi chuyển
sang mục 4.

### Kiểm tra trước khi bàn giao mục 3

- Chạy module cho kết quả khớp toàn bộ đáp án tính tay ở trên.
- Chạy các kiểm tra trực tiếp trên ví dụ lost sales, L = 1 và ngày
  không có nhu cầu. Lượng bán mất hôm trước được ghi riêng theo ngày.
- 100 kịch bản số nguyên với seed 29 thỏa cân bằng nhu cầu, tồn và
  hàng đang về, giữ nguyên đầu vào. Chạy liên tục cho kết quả khớp với
  chạy hai đoạn rồi tiếp tục bằng tồn và đơn đang về của đoạn trước.
- 16 ca đầu vào sai được từ chối, gồm lượng âm, số lẻ, thiếu ngày trong
  lịch đặt, NaN, infinity và đơn có ngày đến trước ngày bắt đầu.
- Kiểm tra tương thích 28 ngày validation của
  `FOODS_1_141_CA_3_evaluation`, đọc từ parquet tập con. Dùng tồn 0, L = 2
  và lịch đặt toàn 0 làm đầu vào minh họa. Ngày và nhu cầu khớp dữ liệu,
  toàn bộ nhu cầu được ghi bán mất đúng như phép tính tay cho trường hợp
  kho trống. Đây là kiểm tra đầu vào của hàm trên dữ liệu M5.
- Ruff đạt trên module mới. Toàn bộ 150 tests hiện có đạt, với một cảnh
  báo deprecation từ SQLAlchemy trong MLflow.

Các kiểm tra riêng của mục 3 được chạy trực tiếp trong môi trường cục
bộ. Bộ pytest cho mô phỏng được bổ sung tại đầu mục 5 theo kế hoạch.
