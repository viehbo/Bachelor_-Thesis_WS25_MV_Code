from datetime import datetime, timezone, timedelta

def safe_value_as_datetime(value):
    """
    TODO
    convert to naive python datetime. Otherwise all before 1970 fails
    maybe there is a better solve
    """
    def one(v):
        if isinstance(v, (int, float)):  # JS ms
            print("FRITZ_1 time is: ", type(v))
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            # allow negative offsets safely:
            dt = epoch + timedelta(milliseconds=v)
            return dt.replace(tzinfo=None)  # Panel expects naive
        # If Panel passes actual datetimes/Timestamps, leave them as-is but naive
        if hasattr(v, "to_pydatetime"):
            print("FRITZ_2 has attribute and type: ", type(v))
            return v.to_pydatetime().replace(tzinfo=None)
        if isinstance(v, datetime):
            print("FRITZ_3 has attribute and type: ", type(v))
            return v.replace(tzinfo=None)
        return v

    if isinstance(value, tuple):
        return one(value[0]), one(value[1])
    return one(value)