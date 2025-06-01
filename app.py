from flask import Flask, render_template, request, redirect, url_for, flash
import os
from FaceRecognition import recognize, load_name_dict
from utils.camera import create_user, train_model
import csv
import pandas as pd

app = Flask(__name__)
app.secret_key = 'face-secret-key'

@app.route('/')
def index():
    return render_template('create_user.html')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        user_id = request.form['user_id']

        create_user(user_id, name)

        flash(f'User {name} added successfully!')
        return redirect('/')
    else:
        return render_template('add_user.html')

@app.route('/train')
def train():
    train_model()
    flash("Model trained successfully.")
    return redirect(url_for('index'))

@app.route('/start_attendance')
def start_attendance():
    names = load_name_dict()
    recognize(names, status="checkin")
    flash("Check-in complete.")
    return redirect(url_for('index'))

@app.route('/end_attendance')
def end_attendance():
    names = load_name_dict()
    recognize(names, status="checkout")
    flash("Check-out complete.")
    return redirect(url_for('index'))

@app.route('/view_attendance')
def view_attendance():
    try:
        df = pd.read_csv('attendance.csv')

        df['Check-in'] = pd.to_datetime(df['Check-in'], errors='coerce')
        df['Check-out'] = pd.to_datetime(df['Check-out'], errors='coerce')

        df['Worked Time'] = (df['Check-out'] - df['Check-in']).dt.total_seconds() / 3600.0
        df['Worked Time'] = df['Worked Time'].apply(lambda x: f"{x:.2f} hrs" if pd.notna(x) else 'N/A')

        df['Check-in'] = df['Check-in'].dt.strftime('%H:%M:%S')
        df['Check-out'] = df['Check-out'].dt.strftime('%H:%M:%S')

        # If 'Date' doesn't exist, extract from Check-in timestamp
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime('today').date()

        grouped = df.groupby('Date')

        grouped_data = []
        for date, group in grouped:
            group.rename(columns={
                'Check-in': 'Check-in Time',
                'Check-out': 'Check-out Time'
            }, inplace=True)

            entries = group[['Name', 'Check-in Time', 'Check-out Time', 'Worked Time']].to_dict(orient='records')
            grouped_data.append({'date': date, 'entries': entries})

        return render_template('view_attendance.html', grouped_data=grouped_data)

    except FileNotFoundError:
        return render_template('view_attendance.html', grouped_data=[], error="Attendance file not found.")


@app.route('/manage_users')
def manage_users():
    users = []
    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                print(f"Processing line: '{line.strip()}'")  # Debug print
                line = line.strip()
                if line and ':' in line:
                    key, val = line.split(":", 1)
                    users.append({'id': key, 'name': val})
                else:
                    print("⚠️ Skipping malformed line:", line)
    return render_template("manage_users.html", users=users)



@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = int(request.form['user_id'])
    name = request.form['name']

    name_dict = {}
    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                print(f"Reading line: '{line.strip()}'")  # Debug print
                line = line.strip()
                if line and ':' in line:
                    k, v = line.split(":", 1)
                    name_dict[int(k)] = v
                else:
                    print("⚠️ Malformed line:", line)

    if user_id in name_dict and name_dict[user_id] == name:
        del name_dict[user_id]

        folder_path = os.path.join("dataset", name)
        if os.path.exists(folder_path):
            import shutil
            shutil.rmtree(folder_path)

        if os.path.exists("attendance.csv"):
            with open("attendance.csv", "r") as f:
                rows = [row for row in csv.reader(f) if row]
            header = rows[0]
            filtered = [row for row in rows[1:] if len(row) > 0 and row[0] != name]
            with open("attendance.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(filtered)

        updated_dict = {}
        for idx, (_, val) in enumerate(sorted(name_dict.items(), key=lambda x: x[1])):
            updated_dict[idx] = val

        with open("names.txt", "w") as f:
            for k, v in updated_dict.items():
                f.write(f"{k}:{v}\n")

        train_model()

        flash(f"User '{name}' deleted and model retrained.")
    else:
        flash("User not found or mismatch.")

    return redirect(url_for('manage_users'))

if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Check-in", "Check-out", "Duration"])
    app.run(debug=True)
