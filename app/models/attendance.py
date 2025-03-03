import datetime

def log_attendance(cursor, conn, user_id):
    timestamp = datetime.datetime.now()
    cursor.execute("INSERT INTO attendance (user_id, timestamp) VALUES (?, ?)", (user_id, timestamp))
    conn.commit()
    return "Attendance logged"
