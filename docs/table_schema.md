erDiagram
    SALES {
        string id
        string item_id
        string dept_id
        string cat_id
        string store_id
        string state_id
        int d_1_den_d_1941 "moi cot la so ban cua 1 ngay"
    }
    CALENDAR {
        string d "d_1, d_2, ..."
        date date
        int wm_yr_wk "ma tuan"
        string event_name_1
        int snap_CA
        int snap_TX
        int snap_WI
    }
    SELL_PRICES {
        string store_id
        string item_id
        int wm_yr_wk
        float sell_price
    }
    SALES }o--|| CALENDAR : "ten cot d_x = cot d"
    CALENDAR }o--o{ SELL_PRICES : "wm_yr_wk (1 tuan gom 7 ngay)"
    SALES }o--o{ SELL_PRICES : "store_id + item_id"