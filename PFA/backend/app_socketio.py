"""
Flask API + SocketIO — IDS using Machine Learning
Phase 6: REST API + WebSocket Real-time Detection

REST Endpoints:
  GET  /health         → is the API running?
  GET  /classes        → names of the 7 classes
  GET  /features       → names of the 43 features
  GET  /stats          → model statistics
  POST /predict        → single connection → prediction
  POST /predict/batch  → multiple connections

Auth (JWT):
  POST /auth/register        → create a 'viewer' account
  POST /auth/login           → returns a JWT token (valid 8h)
  GET  /auth/me               → info about the logged-in user
  GET  /alerts/history         → requires token (viewer or admin)
  GET  /alerts/stats           → requires token (viewer or admin)
  DELETE /alerts/<id>          → requires token + role='admin'

SocketIO Events:
  Client → Server:
    'connect'          → client connected
    'disconnect'       → client disconnected

  Server → Client (emit):
    'alert'            → attack detected → Dashboard
    'stats_update'     → update statistics
    'new_prediction'   → every prediction (alert or not)
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import pickle
import json
import time
import os
from datetime import datetime, timedelta
from functools import wraps

# --- Auth (JWT + password hashing) ---
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# --- Database (SQLite + SQLAlchemy) ---
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime, Boolean,
    ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base, relationship

# ============================================================
# Step 1 — Init Flask + SocketIO
# ============================================================
# SocketIO(app, cors_allowed_origins="*"):
#   cors_allowed_origins="*" → allows any client
#   (browser, Dashboard, simulation script...)
#   to connect via WebSocket
#
# async_mode='threading':
#   lets Flask handle REST and WebSocket at the same time
#   without blocking either one

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ids-secret-key-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

print("=" * 55)
print("IDS Flask API + SocketIO — Starting...")
print("=" * 55)

# ============================================================
# Step 2 — Load Model and Metadata
# ============================================================
print("\nLoading model and metadata...")

with open('ids_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

with open('selected_features.json', 'r') as f:
    feature_names = json.load(f)

with open('evaluation_report.json', 'r') as f:
    eval_report = json.load(f)

print(f"✅ Model loaded: XGBoost (multi:softprob)")
print(f"✅ Classes: {class_names}")
print(f"✅ Features: {len(feature_names)}")
print(f"✅ SocketIO: enabled")

# ============================================================
# Step 2b — Database (SQLite + SQLAlchemy ORM)
# ============================================================
# SQLite here is just for dev/PFA (a single .db file, no server).
# In production, you'd only need to swap the engine for PostgreSQL:
#   create_engine("postgresql://user:pass@host:5432/dbname")
# ...the rest of the code (models, queries) works unchanged,
# since SQLAlchemy is the layer that abstracts the DB away.

DB_PATH = "sqlite:///ids_alerts.db"
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()


class ClasseAttaque(Base):
    """
    Table 'classes_attaque' — reference table for the model's 7 classes (lookup table).
    MCD association: Alert (0,n) —CONCERNS→ (1,1) ClasseAttaque
    """
    __tablename__ = "classes_attaque"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    nom_classe       = Column(String(50), unique=True, nullable=False)
    niveau_severite  = Column(String(20), nullable=False)
    couleur          = Column(String(10), nullable=True)

    def to_dict(self):
        return {
            "id": self.id, "nom_classe": self.nom_classe,
            "niveau_severite": self.niveau_severite, "couleur": self.couleur
        }


class ModeleML(Base):
    """
    Table 'modeles_ml' — ML model versions used to make predictions.
    MCD association: Alert (0,n) —GENERATED_BY→ (1,1) ModeleML
    """
    __tablename__ = "modeles_ml"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    nom_modele       = Column(String(80), nullable=False)
    type_algorithme  = Column(String(50), nullable=False)
    nb_features      = Column(Integer, nullable=False)
    nb_classes       = Column(Integer, nullable=False)
    date_ajout       = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "nom_modele": self.nom_modele,
            "type_algorithme": self.type_algorithme,
            "nb_features": self.nb_features, "nb_classes": self.nb_classes,
            "date_ajout": self.date_ajout.isoformat()
        }


class Alert(Base):
    """Table 'alerts' — every detected attack (is_alert=True) is stored here"""
    __tablename__ = "alerts"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    prediction       = Column(String(50),  nullable=False)   # class name (DDoS, Bot...)
    confidence       = Column(Float,       nullable=False)   # prediction probability
    severity         = Column(String(20),  nullable=False)   # NONE/MEDIUM/HIGH/CRITICAL
    probabilities    = Column(Text,        nullable=True)    # JSON string {class: proba}
    response_time_ms = Column(Float,       nullable=True)
    source           = Column(String(30),  default="predict")  # predict/predict_batch/predict_flow
    timestamp        = Column(DateTime,    default=datetime.utcnow, index=True)

    # --- FKs (MCD associations) ---
    id_classe          = Column(Integer, ForeignKey("classes_attaque.id"), nullable=True)
    id_modele           = Column(Integer, ForeignKey("modeles_ml.id"), nullable=True)
    acquittee            = Column(Boolean, default=False)
    acquittee_par_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    date_acquittement     = Column(DateTime, nullable=True)

    classe        = relationship("ClasseAttaque")
    modele        = relationship("ModeleML")
    acquitte_par  = relationship("User", foreign_keys=[acquittee_par_id])

    def to_dict(self):
        return {
            "id":                self.id,
            "prediction":        self.prediction,
            "confidence":        self.confidence,
            "severity":          self.severity,
            "probabilities":     json.loads(self.probabilities) if self.probabilities else None,
            "response_time_ms":  self.response_time_ms,
            "source":            self.source,
            "timestamp":         self.timestamp.isoformat(),
            "classe_attaque":    self.classe.to_dict() if self.classe else None,
            "modele":            self.modele.nom_modele if self.modele else None,
            "acquittee":         self.acquittee,
            "acquittee_par":     self.acquitte_par.username if self.acquitte_par else None,
            "date_acquittement": self.date_acquittement.isoformat() if self.date_acquittement else None,
        }


class User(Base):
    """Table 'users' — for JWT auth (admin/viewer)"""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), default="viewer")   # admin | viewer
    created_at    = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "username":   self.username,
            "role":       self.role,
            "created_at": self.created_at.isoformat()
        }


# Creates the tables if they don't exist yet (idempotent — safe to run every time)
Base.metadata.create_all(engine)
print(f"✅ Database: SQLite ready (ids_alerts.db) — tables: alerts, users, classes_attaque, modeles_ml")

# ============================================================
# Step 2c — Authentication (JWT + admin/viewer roles)
# ============================================================
# How it works:
#   1. POST /auth/register → creates a new user (role='viewer' by default)
#   2. POST /auth/login    → verifies username/password → returns a JWT token
#   3. Protected endpoints  → must send header:
#        Authorization: Bearer <token>
#   4. @token_required      → blocks access if the token isn't valid
#   5. @admin_required      → blocks access if role != 'admin'
#
# Passwords are never stored in plaintext in the DB — only their hash is stored
# (generate_password_hash / check_password_hash from werkzeug).

JWT_SECRET     = (app.config['SECRET_KEY'] * 3)[:48]  # >=32 bytes to avoid InsecureKeyLengthWarning
JWT_ALGORITHM  = "HS256"
JWT_EXP_HOURS  = 8


def generate_token(user):
    """Builds a JWT token containing username, role, and expiration date"""
    payload = {
        "username": user.username,
        "role":     user.role,
        "exp":      datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS),
        "iat":      datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def token_required(f):
    """Decorator: requires a valid token in the Authorization header"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing token. Add header: Authorization: Bearer <token>"}), 401

        token = auth_header.split(' ', 1)[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        g.current_user = {"username": payload["username"], "role": payload["role"]}
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator: requires token_required + role == 'admin'"""
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if g.current_user["role"] != "admin":
            return jsonify({"error": "Access restricted to admins"}), 403
        return f(*args, **kwargs)
    return decorated


def seed_default_admin():
    """If the DB is empty (first run), creates a default admin so you can log in"""
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            default_admin = User(
                username=os.environ.get("IDS_ADMIN_USER", "admin"),
                password_hash=generate_password_hash(os.environ.get("IDS_ADMIN_PASSWORD", "admin123")),
                role="admin"
            )
            db.add(default_admin)
            db.commit()
            print("=" * 55)
            print("👤 Default admin account created:")
            print(f"   username: {default_admin.username}")
            print(f"   password: {os.environ.get('IDS_ADMIN_PASSWORD', 'admin123')}")
            print("   ⚠️  Change this password before your first production login")
            print("=" * 55)
    finally:
        db.close()


seed_default_admin()

# Stats + History
api_stats = {
    "total_requests": 0,
    "total_alerts": 0,
    "predictions_per_class": {cls: 0 for cls in class_names},
    "start_time": datetime.now().isoformat()
}

# Last 50 alerts (for the Dashboard)
alerts_history = []

SEVERITY = {
    "Benign":       "NONE",
    "Bot":          "HIGH",
    "BruteForce":   "MEDIUM",
    "DDoS":         "CRITICAL",
    "DoS":          "HIGH",
    "Infiltration": "CRITICAL",
    "WebAttack":    "MEDIUM"
}

SEVERITY_COLOR = {
    "NONE":     "#2ecc71",   # green
    "MEDIUM":   "#f39c12",   # orange
    "HIGH":     "#e67e22",   # dark orange
    "CRITICAL": "#e74c3c"    # red
}

# ============================================================
# Step 2d — Seeding the reference tables (ClasseAttaque, ModeleML)
# ============================================================
CLASS_ID_MAP   = {}   # {"DDoS": 4, "Bot": 2, ...} — used by emit_prediction
ACTIVE_MODEL_ID = None


def seed_classes_attaque():
    """Fills 'classes_attaque' once (idempotent) from class_names.json + SEVERITY"""
    global CLASS_ID_MAP
    db = SessionLocal()
    try:
        if db.query(ClasseAttaque).count() == 0:
            for cls in class_names:
                sev = SEVERITY.get(cls, "UNKNOWN")
                db.add(ClasseAttaque(nom_classe=cls, niveau_severite=sev,
                                      couleur=SEVERITY_COLOR.get(sev)))
            db.commit()
        CLASS_ID_MAP = {c.nom_classe: c.id for c in db.query(ClasseAttaque).all()}
    finally:
        db.close()


def seed_modele_actif():
    """Registers (once) the currently loaded model as the 'active model'"""
    global ACTIVE_MODEL_ID
    db = SessionLocal()
    try:
        m = db.query(ModeleML).filter_by(nom_modele="ids_model_xgb.pkl").first()
        if not m:
            m = ModeleML(
                nom_modele="ids_model_xgb.pkl",
                type_algorithme="XGBoost (multi:softprob)",
                nb_features=len(feature_names),
                nb_classes=len(class_names)
            )
            db.add(m)
            db.commit()
        ACTIVE_MODEL_ID = m.id
    finally:
        db.close()


seed_classes_attaque()
seed_modele_actif()
print(f"✅ Reference data: {len(CLASS_ID_MAP)} attack classes, active model #{ACTIVE_MODEL_ID}")

# Connected clients counter
connected_clients = 0


# ============================================================
# Step 3 — SocketIO Events
# ============================================================

@socketio.on('connect')
def handle_connect():
    """Client connected — shown in the terminal"""
    global connected_clients
    connected_clients += 1
    print(f"✅ Client connected: {request.sid} (total: {connected_clients})")

    # Send initial stats to the new client
    emit('stats_update', {
        "total_requests": api_stats["total_requests"],
        "total_alerts":   api_stats["total_alerts"],
        "predictions_per_class": api_stats["predictions_per_class"],
        "connected_clients": connected_clients
    })

    # Send alert history to the new client
    emit('alerts_history', {"alerts": alerts_history[-20:]})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"Client disconnected: {request.sid} (total: {connected_clients})")


# ============================================================
# Helper function — emit_prediction
# ============================================================
# This function runs after every prediction:
#   1. Updates stats
#   2. If it's an alert → emits the 'alert' event to all clients
#   3. Emits the 'new_prediction' event (alert or not)
#   4. Updates 'stats_update' for all clients
#
# socketio.emit() (without 'broadcast') → all connected clients
# emit() (with broadcast=True)          → same result

def emit_prediction(prediction_name, confidence, is_alert,
                    severity, probabilities, response_time_ms, source="predict"):
    """Helper: updates stats, emits SocketIO events, and stores the alert in the DB"""

    # Update stats
    api_stats["total_requests"] += 1
    api_stats["predictions_per_class"][prediction_name] += 1
    if is_alert:
        api_stats["total_alerts"] += 1

    # Build prediction object
    prediction_data = {
        "prediction":    prediction_name,
        "confidence":    round(confidence, 4),
        "alert":         is_alert,
        "severity":      severity,
        "color":         SEVERITY_COLOR.get(severity, "#95a5a6"),
        "probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(class_names, probabilities)
        },
        "response_time_ms": response_time_ms,
        "timestamp":     datetime.now().isoformat()
    }

    # Emit 'new_prediction' → Dashboard shows every prediction
    socketio.emit('new_prediction', prediction_data)

    # If it's an alert → emit 'alert' event (Dashboard shows a notification)
    if is_alert:
        alert_data = {
            **prediction_data,
            "alert_id": api_stats["total_alerts"]
        }
        socketio.emit('alert', alert_data)
        alerts_history.append(alert_data)
        # Keep only the last 100 alerts in memory
        if len(alerts_history) > 100:
            alerts_history.pop(0)

        # --- Store in SQLite (persistent, survives restarts unlike memory) ---
        db = SessionLocal()
        try:
            db_alert = Alert(
                prediction=prediction_name,
                confidence=confidence,
                severity=severity,
                probabilities=json.dumps(prediction_data["probabilities"]),
                response_time_ms=response_time_ms,
                source=source,
                id_classe=CLASS_ID_MAP.get(prediction_name),
                id_modele=ACTIVE_MODEL_ID
            )
            db.add(db_alert)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"⚠️  DB error (alert could not be saved): {e}")
        finally:
            db.close()

        print(f"🚨 ALERT #{api_stats['total_alerts']}: "
            f"{prediction_name} (conf: {confidence:.2%}) "
            f"[{severity}]")

        # Refresh stats immediately on every alert (not just every 10 requests)
        socketio.emit('stats_update', {
            "total_requests":        api_stats["total_requests"],
            "total_alerts":          api_stats["total_alerts"],
            "predictions_per_class": api_stats["predictions_per_class"],
            "connected_clients":     connected_clients
        })

    # Emit stats update every 10 requests
    if api_stats["total_requests"] % 10 == 0:
        socketio.emit('stats_update', {
            "total_requests":        api_stats["total_requests"],
            "total_alerts":          api_stats["total_alerts"],
            "predictions_per_class": api_stats["predictions_per_class"],
            "connected_clients":     connected_clients
        })

    return prediction_data


# ============================================================
# Step 4 — REST Endpoints
# ============================================================

@app.route('/auth/register', methods=['POST'])
def auth_register():
    """
    Creates a 'viewer' account (the admin role can only be created via
    seed_default_admin or directly in the DB, never through this public
    route — for security).
    Body: {"username": "...", "password": "..."}
    """
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400

    db = SessionLocal()
    try:
        if db.query(User).filter_by(username=username).first():
            return jsonify({"error": "this username already exists"}), 409

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role="viewer"
        )
        db.add(user)
        db.commit()
        return jsonify({"message": "Account created", "user": user.to_dict()}), 201
    finally:
        db.close()


@app.route('/auth/login', methods=['POST'])
def auth_login():
    """
    Body: {"username": "...", "password": "..."}
    Returns a JWT token valid for 8h if the credentials are correct.
    """
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "incorrect username or password"}), 401

        token = generate_token(user)
        return jsonify({
            "token": token,
            "expires_in_hours": JWT_EXP_HOURS,
            "user": user.to_dict()
        })
    finally:
        db.close()


@app.route('/auth/me', methods=['GET'])
@token_required
def auth_me():
    """Returns the logged-in user (so the frontend can verify the session)"""
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=g.current_user["username"]).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"user": user.to_dict()})
    finally:
        db.close()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "IDS API + SocketIO running fine",
        "model": "XGBClassifier",
        "n_classes": len(class_names),
        "n_features": len(feature_names),
        "websocket": "enabled",
        "connected_clients": connected_clients,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    classes_info = []
    for i, cls in enumerate(class_names):
        severity = SEVERITY.get(cls, "UNKNOWN")
        classes_info.append({
            "id":       i,
            "name":     cls,
            "severity": severity,
            "color":    SEVERITY_COLOR.get(severity, "#95a5a6")
        })
    return jsonify({"classes": classes_info, "total": len(class_names)})


@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({"features": feature_names, "total": len(feature_names)})


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "api_stats": {
            "total_requests":        api_stats["total_requests"],
            "total_alerts":          api_stats["total_alerts"],
            "predictions_per_class": api_stats["predictions_per_class"],
            "uptime_since":          api_stats["start_time"],
            "connected_clients":     connected_clients
        },
        "model_performance": {
            "macro_f1":     eval_report["metrics"]["macro_f1"],
            "accuracy":     eval_report["metrics"]["accuracy"],
            "macro_recall": eval_report["metrics"]["macro_recall"],
            "per_class_f1": eval_report["per_class_f1"]
        },
        "recent_alerts": alerts_history[-10:]
    })
@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Last N alerts (in-memory, live — for the real-time Dashboard)"""
    n = int(request.args.get('n', 20))
    return jsonify({
        "alerts": alerts_history[-n:],
        "total":  len(alerts_history)
    })


@app.route('/alerts/history', methods=['GET'])
@token_required
def get_alerts_history():
    """
    Persistent history (SQLite) — not wiped on restart.
    Query params:
      severity   → filter (MEDIUM/HIGH/CRITICAL)
      prediction → filter by class name (DDoS, Bot...)
      limit      → default 50, max 500
      offset     → pagination (default 0)
    """
    severity   = request.args.get('severity')
    prediction = request.args.get('prediction')
    limit      = min(int(request.args.get('limit', 50)), 500)
    offset     = int(request.args.get('offset', 0))

    db = SessionLocal()
    try:
        query = db.query(Alert)
        if severity:
            query = query.filter(Alert.severity == severity.upper())
        if prediction:
            query = query.filter(Alert.prediction == prediction)

        total = query.count()
        rows = (query.order_by(Alert.timestamp.desc())
                     .offset(offset).limit(limit).all())

        return jsonify({
            "alerts": [r.to_dict() for r in rows],
            "total":  total,
            "limit":  limit,
            "offset": offset
        })
    finally:
        db.close()


@app.route('/alerts/stats', methods=['GET'])
@token_required
def get_alerts_stats():
    """
    Aggregated statistics from the database (the full history, not just memory).
    """
    db = SessionLocal()
    try:
        total_alerts = db.query(Alert).count()

        per_severity = dict(
            db.query(Alert.severity, func.count(Alert.id))
              .group_by(Alert.severity).all()
        )
        per_class = dict(
            db.query(Alert.prediction, func.count(Alert.id))
              .group_by(Alert.prediction).all()
        )
        per_source = dict(
            db.query(Alert.source, func.count(Alert.id))
              .group_by(Alert.source).all()
        )
        avg_confidence = db.query(func.avg(Alert.confidence)).scalar()
        avg_response_ms = db.query(func.avg(Alert.response_time_ms)).scalar()

        last_alert = (db.query(Alert)
                        .order_by(Alert.timestamp.desc())
                        .first())

        return jsonify({
            "total_alerts_db":   total_alerts,
            "per_severity":      per_severity,
            "per_class":         per_class,
            "per_source":        per_source,
            "avg_confidence":    round(avg_confidence, 4) if avg_confidence else None,
            "avg_response_ms":   round(avg_response_ms, 2) if avg_response_ms else None,
            "last_alert":        last_alert.to_dict() if last_alert else None
        })
    finally:
        db.close()


@app.route('/alerts/<int:alert_id>', methods=['DELETE'])
@admin_required
def delete_alert(alert_id):
    """
    Deletes an alert from the history (DB). Admins only —
    demonstrates role-based access control (RBAC).
    """
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter_by(id=alert_id).first()
        if not alert:
            return jsonify({"error": "Alert not found"}), 404
        db.delete(alert)
        db.commit()
        return jsonify({"message": f"Alert #{alert_id} deleted", "by": g.current_user["username"]})
    finally:
        db.close()


# ============================================================
# Step 5 — /predict (Main endpoint + SocketIO emit)
# ============================================================

@app.route('/alerts/<int:alert_id>/acquitter', methods=['POST'])
@token_required
def acquitter_alert(alert_id):
    """
    Marks an alert as acknowledged (reviewed/handled) by the logged-in user.
    Accessible to any authenticated user (admin or viewer).
    → Implements the MCD association: User (0,1) —ACKNOWLEDGES→ (0,n) Alert
    """
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter_by(id=alert_id).first()
        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        user = db.query(User).filter_by(username=g.current_user["username"]).first()
        alert.acquittee = True
        alert.acquittee_par_id = user.id if user else None
        alert.date_acquittement = datetime.utcnow()
        db.commit()
        return jsonify({"message": f"Alert #{alert_id} acknowledged", "alert": alert.to_dict()})
    finally:
        db.close()


@app.route('/auth/users', methods=['GET'])
@admin_required
def list_users():
    """List of accounts — admins only (RBAC demo)"""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return jsonify({"users": [u.to_dict() for u in users], "total": len(users)})
    finally:
        db.close()


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction + SocketIO emit"""
    t0 = time.time()
    data = request.get_json()

    if not data or 'features' not in data:
        return jsonify({
            "error": "'features' is required in the request body",
            "example": {"features": [0.0] * len(feature_names)}
        }), 400

    features = data['features']

    if len(features) != len(feature_names):
        return jsonify({
            "error": f"Expected {len(feature_names)} features, got {len(features)}"
        }), 400

    X = np.array([features])
    prediction_idx  = int(model.predict(X)[0])
    prediction_name = class_names[prediction_idx]
    probabilities   = model.predict_proba(X)[0]
    confidence      = float(probabilities[prediction_idx])
    is_alert        = prediction_name != "Benign"
    severity        = SEVERITY.get(prediction_name, "UNKNOWN")
    response_time_ms = round((time.time() - t0) * 1000, 2)
    # Emit via SocketIO → Dashboard updates in real time
    prediction_data = emit_prediction(
        prediction_name, confidence, is_alert,
        severity, probabilities, response_time_ms,
        source="predict"
    )
    return jsonify({
        **prediction_data,
        "prediction_id": prediction_idx
    })


# ============================================================
# Step 6 — /predict/batch + SocketIO emit
# ============================================================

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch predictions + SocketIO emit"""
    t0 = time.time()
    data = request.get_json()
    if not data or 'connections' not in data:
        return jsonify({"error": "'connections' is required in the request body"}), 400
    connections = data['connections']
    if len(connections) == 0:
        return jsonify({"error": "list is empty"}), 400
    if len(connections) > 1000:
        return jsonify({"error": "Maximum 1000 connections"}), 400
    X_batch = []
    errors  = []
    for i, conn in enumerate(connections):
        if 'features' not in conn:
            errors.append(f"Connection {i}: 'features' is required")
            continue
        if len(conn['features']) != len(feature_names):
            errors.append(f"Connection {i}: wrong features count")
            continue
        X_batch.append(conn['features'])
    if not X_batch:
        return jsonify({"errors": errors}), 400
    X = np.array(X_batch)
    predictions_idx   = model.predict(X)
    probabilities_all = model.predict_proba(X)
    results  = []
    n_alerts = 0

    for i, (pred_idx, probs) in enumerate(zip(predictions_idx, probabilities_all)):
        pred_idx  = int(pred_idx)
        pred_name = class_names[pred_idx]
        conf      = float(probs[pred_idx])
        is_alert  = pred_name != "Benign"
        severity  = SEVERITY.get(pred_name, "UNKNOWN")
        rt_ms     = round((time.time() - t0) * 1000, 2)

        if is_alert:
            n_alerts += 1

        # Emit every prediction via SocketIO
        emit_prediction(pred_name, conf, is_alert, severity, probs, rt_ms,
                        source="predict_batch")

        results.append({
            "index":      i,
            "prediction": pred_name,
            "confidence": round(conf, 4),
            "alert":      is_alert,
            "severity":   severity,
            "color":      SEVERITY_COLOR.get(severity, "#95a5a6")
        })

    return jsonify({
        "results": results,
        "summary": {
            "total":      len(results),
            "alerts":     n_alerts,
            "benign":     len(results) - n_alerts,
            "alert_rate": round(n_alerts / len(results), 4)
        },
        "response_time_ms": round((time.time() - t0) * 1000, 2),
        "timestamp": datetime.now().isoformat(),
        "errors": errors if errors else None
    })
# ============================================================
# Step 7 — /predict/flow (cicflowmeter endpoint)
# ============================================================
# This endpoint is meant to be called directly by cicflowmeter
#
# cicflowmeter sends JSON like:
# {
#   "dst_port": 80,
#   "flow_duration": 1200,
#   "tot_fwd_pkts": 5000,
#   ...
# }
#
# /predict expects a list [val1, val2, ...]
# /predict/flow expects a dict with keys
#
# Important mapping:
#   cicflowmeter keys → our feature_names (in the same order)

# cicflowmeter column names → our feature_names
CICFLOW_MAPPING = {
    "dst_port":           "Dst Port",
    "protocol":           "Protocol",
    "flow_duration":      "Flow Duration",
    "tot_fwd_pkts":       "Tot Fwd Pkts",
    "tot_bwd_pkts":       "Tot Bwd Pkts",
    "totlen_fwd_pkts":    "TotLen Fwd Pkts",
    "totlen_bwd_pkts":    "TotLen Bwd Pkts",
    "fwd_pkt_len_max":    "Fwd Pkt Len Max",
    "fwd_pkt_len_min":    "Fwd Pkt Len Min",
    "fwd_pkt_len_mean":   "Fwd Pkt Len Mean",
    "fwd_pkt_len_std":    "Fwd Pkt Len Std",
    "bwd_pkt_len_max":    "Bwd Pkt Len Max",
    "bwd_pkt_len_min":    "Bwd Pkt Len Min",
    "bwd_pkt_len_mean":   "Bwd Pkt Len Mean",
    "bwd_pkt_len_std":    "Bwd Pkt Len Std",
    "flow_byts_s":        "Flow Byts/s",
    "flow_pkts_s":        "Flow Pkts/s",
    "flow_iat_mean":      "Flow IAT Mean",
    "flow_iat_std":       "Flow IAT Std",
    "flow_iat_max":       "Flow IAT Max",
    "flow_iat_min":       "Flow IAT Min",
    "fwd_iat_tot":        "Fwd IAT Tot",
    "fwd_iat_mean":       "Fwd IAT Mean",
    "fwd_iat_std":        "Fwd IAT Std",
    "fwd_iat_max":        "Fwd IAT Max",
    "fwd_iat_min":        "Fwd IAT Min",
    "bwd_iat_tot":        "Bwd IAT Tot",
    "bwd_iat_mean":       "Bwd IAT Mean",
    "bwd_iat_std":        "Bwd IAT Std",
    "bwd_iat_max":        "Bwd IAT Max",
    "bwd_iat_min":        "Bwd IAT Min",
    "fwd_psh_flags":      "Fwd PSH Flags",
    "fwd_urg_flags":      "Fwd URG Flags",
    "fwd_header_len":     "Fwd Header Len",
    "bwd_header_len":     "Bwd Header Len",
    "fwd_pkts_s":         "Fwd Pkts/s",
    "bwd_pkts_s":         "Bwd Pkts/s",
    "pkt_len_min":        "Pkt Len Min",
    "pkt_len_max":        "Pkt Len Max",
    "pkt_len_mean":       "Pkt Len Mean",
    "pkt_len_std":        "Pkt Len Std",
    "pkt_len_var":        "Pkt Len Var",
}

@app.route('/predict/flow', methods=['POST'])
def predict_flow():
    """
    Special endpoint for cicflowmeter
    Accepts a JSON dict with keys (not a list)
    and converts it → a list in feature_names order
    """
    t0   = time.time()
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Empty or invalid JSON body",
            "example": {
                "dst_port": 80,
                "flow_duration": 1200,
                "tot_fwd_pkts": 5000,
                "flow_pkts_s": 4166.0
            }
        }), 400

    # Convert cicflowmeter keys → feature values in feature_names order
    # any feature not present in data → 0.0 (default)
    features = []
    missing_features = []

    for feat_name in feature_names:
        # Find the cicflowmeter key that corresponds to this feature
        cic_key = None
        for cic, feat in CICFLOW_MAPPING.items():
            if feat == feat_name:
                cic_key = cic
                break
        if cic_key and cic_key in data:
            try:
                val = float(data[cic_key])
                # Handle NaN/Inf
                if val != val or abs(val) == float('inf'):
                    val = 0.0
                features.append(val)
            except (TypeError, ValueError):
                features.append(0.0)
        else:
            # Feature not present → 0.0
            features.append(0.0)
            missing_features.append(feat_name)

    # Prediction
    X               = np.array([features])
    prediction_idx  = int(model.predict(X)[0])
    prediction_name = class_names[prediction_idx]
    probabilities   = model.predict_proba(X)[0]
    confidence      = float(probabilities[prediction_idx])
    is_alert        = prediction_name != "Benign"
    severity        = SEVERITY.get(prediction_name, "UNKNOWN")
    response_time_ms = round((time.time() - t0) * 1000, 2)

    # Emit via SocketIO → Dashboard updates in real time
    prediction_data = emit_prediction(
        prediction_name, confidence, is_alert,
        severity, probabilities, response_time_ms,
        source="predict_flow"
    )

    return jsonify({
        **prediction_data,
        "prediction_id":    prediction_idx,
        "source":           "cicflowmeter",
        "missing_features": missing_features if missing_features else None,
        "note": "missing features set to 0.0 by default"
    })
# ============================================================

# ============================================================
# socketio.run() instead of app.run()
# starts Flask and the SocketIO server at the same time

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("IDS API + SocketIO Ready!")
    print("=" * 55)
    print(f"  REST API:   http://localhost:5000")
    print(f"  WebSocket:  ws://localhost:5000")
    print(f"  Health:     http://localhost:5000/health")
    print(f"  Stats:      http://localhost:5000/stats")
    print(f"  Alerts:     http://localhost:5000/alerts")
    print(f"  History:    http://localhost:5000/alerts/history")
    print(f"  DB Stats:   http://localhost:5000/alerts/stats")
    print(f"  Predict:    POST http://localhost:5000/predict")
    print(f"  Batch:      POST http://localhost:5000/predict/batch")
    print("\nPress Ctrl+C to stop the API")
    print("=" * 55 + "\n")

import os

port = int(os.environ.get("PORT", 5000))

socketio.run(
    app,
    host="0.0.0.0",
    port=port,
    debug=False,
    allow_unsafe_werkzeug=True
)