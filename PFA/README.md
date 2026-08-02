# IDS 2025 — Détection d'alertes réseau en temps réel

Projet PFA · 1ère année Cycle Ingénieur · ENSIASD Taroudant

Structure alignée sur l'architecture (couches ①→④, la couche ⑤ Infrastructure/Déploiement est volontairement laissée pour plus tard) :

```
pfa_ids_project/
├── backend/              # Couche ③ — API + WebSocket + DB
│   ├── app_socketio.py       Flask + SocketIO + SQLite (SQLAlchemy)
│   ├── ids_model_xgb.pkl     Modèle entraîné (XGBoost, 7 classes, 43 features)
│   ├── class_names.json      Noms des 7 classes
│   ├── selected_features.json  Ordre des 43 features attendues par le modèle
│   ├── evaluation_report.json  Métriques d'évaluation (PLACEHOLDER — à remplacer)
│   └── requirements.txt
├── simulator/            # Couche "Simulation" — rejoue le dataset
│   ├── simulate_traffic.py
│   └── requirements.txt
└── frontend/             # Couche ④ — Dashboard temps réel
    └── dashboard.html        Fichier unique, aucun build nécessaire
```

## ⚠️ 3 choses à vérifier avant de considérer le pipeline "final"

1. **`class_names.json`** — l'ordre `["Benign","Bot","BruteForce","DDoS","DoS","Infiltration","WebAttack"]`
   est une supposition (ordre alphabétique / cohérent avec le dict `SEVERITY` déjà dans le code).
   Si ton `LabelEncoder` d'entraînement a utilisé un autre ordre, les prédictions seront décalées.
   → Vérifie dans ton notebook ML : `label_encoder.classes_` ou l'équivalent.

2. **`selected_features.json`** — liste de 43 noms de features "façon CICFlowMeter", construite pour
   correspondre au nombre exact attendu par le modèle (`n_features_in_ = 43`). Si ton vrai pipeline de
   feature engineering a gardé un ordre/nommage différent, remplace ce fichier par la vraie liste
   exportée depuis ton notebook (`X_train.columns.tolist()`).

3. **`evaluation_report.json`** — actuellement des zéros. Remplace par les vraies métriques
   (`accuracy`, `macro_f1`, `macro_recall`, `per_class_f1`) issues de ton `classification_report`.

## Lancer le projet en local

### 1) Backend (API + WebSocket + DB)
```bash
cd backend
pip install -r requirements.txt
python app_socketio.py
```
Démarre sur `http://localhost:5000`. La base SQLite (`ids_alerts.db`) est créée automatiquement
au premier lancement (tables `alerts`, `users`).

Endpoints clés :
- `GET  /health`, `/classes`, `/features`, `/stats`
- `POST /predict`, `/predict/batch`, `/predict/flow`
- `GET  /alerts` (live, mémoire), `/alerts/history` (persistant, filtrable), `/alerts/stats` (agrégats DB)

### 2) Dashboard
Ouvre `frontend/dashboard.html` directement dans le navigateur (double-clic, pas de serveur requis).
Vérifie que l'URL affichée en haut correspond à ton backend (`http://localhost:5000`), clique **Connecter**.

### 3) Simulateur (rejoue le dataset IDS 2025)
```bash
cd simulator
pip install -r requirements.txt
python simulate_traffic.py --csv chemin/vers/ton_dataset.csv --speed 3
```
Options utiles :
- `--label-col Label` → si ton CSV a une colonne de vrai label, le simulateur calcule l'accuracy en direct
- `--loop` → rejoue le dataset en boucle infinie
- `--shuffle` → mélange les lignes avant de les envoyer
- `--limit 500` → limite le nombre de lignes rejouées

## Ce qui reste (Couche ⑤ — pour plus tard)
Docker, GitHub CI/CD, déploiement (Render/Railway). Rien n'a été touché ici, comme convenu.

## 🔐 Authentification (JWT + rôles admin/viewer)

Ajoutée pour couvrir la case "Auth utilisateur — JWT · Login · Rôles admin/viewer" de la couche ④.

**Compte admin par défaut** (créé automatiquement au 1er lancement du backend) :
```
username: admin
password: admin123
```
⚠️ À changer avant toute démo publique ou déploiement.

**Flux :**
1. `POST /auth/register` → crée un compte `viewer`
2. `POST /auth/login` → renvoie un token JWT (valide 8h)
3. Le frontend stocke le token et l'envoie dans le header `Authorization: Bearer <token>`
4. Routes protégées : `GET /alerts/history`, `GET /alerts/stats` (tout utilisateur connecté),
   `DELETE /alerts/<id>` et `GET /auth/users` (admin uniquement)
5. `POST /alerts/<id>/acquitter` (tout utilisateur connecté) — marque une alerte comme traitée

Le dashboard (`frontend/dashboard.html`) démarre sur un écran de connexion ; une fois connecté,
le badge en haut à droite affiche le nom d'utilisateur et le rôle, et un bouton "supprimer" n'apparaît
sur les lignes de l'historique que pour les comptes `admin` (démonstration RBAC).

## 🗂️ Modèle de données (MCD) — 4 entités, 3 associations

```
UTILISATEUR (0,1) ──acquitte──> (0,n) ALERTE
ALERTE      (0,n) ──concerne──> (1,1) CLASSE_ATTAQUE
ALERTE      (0,n) ──généréé_par─> (1,1) MODELE_ML
```

| Entité | Attributs clés |
|---|---|
| **Utilisateur** | id (PK), username, password_hash, role, date_creation |
| **Alerte** | id (PK), prediction, confidence, severity, probabilities, response_time_ms, source, timestamp, id_classe (FK), id_modele (FK), acquittee, acquittee_par_id (FK), date_acquittement |
| **Classe_attaque** | id (PK), nom_classe, niveau_severite, couleur — référentiel des 7 classes du modèle |
| **Modele_ML** | id (PK), nom_modele, type_algorithme, nb_features, nb_classes, date_ajout |

Note : l'association **Acquitte** porte un attribut (`date_acquittement`) — pour rester simple,
il est matérialisé directement comme colonne sur `Alerte` plutôt que comme table d'association
séparée (choix courant en MCD/MLD quand l'association est 0,1–0,n avec peu d'attributs).

Ce schéma est implémenté avec SQLAlchemy dans `backend/app_socketio.py` (classes `User`, `Alert`,
`ClasseAttaque`, `ModeleML`) et testé de bout en bout (login → predict → alerte liée à sa classe et
son modèle → acquittement → suppression admin).

## ✅ Résultats des tests (fait de bout en bout, pas juste en théorie)

Tout le pipeline a été testé avec un vrai serveur qui tourne + un vrai client Socket.IO (pas de mock) :

| Test | Résultat |
|---|---|
| `/health`, `/classes`, `/features` | ✅ OK |
| `/auth/register`, `/auth/login`, `/auth/me` | ✅ OK |
| WebSocket réel (transport `websocket`) — `Client connecté` dans les logs | ✅ OK |
| `/predict` → alerte → sauvegardée en DB avec `classe_attaque` + `modele` liés | ✅ OK |
| `/alerts/history`, `/alerts/stats` (protégés JWT) | ✅ OK |
| `POST /alerts/<id>/acquitter` | ✅ OK |
| RBAC : viewer bloqué sur `DELETE /alerts/<id>` et `GET /auth/users` (403) | ✅ OK |
| Accès sans token → 401 | ✅ OK |
| `simulate_traffic.py` (CSV → `/predict`, avec et sans `--label-col`) | ✅ OK |

**Astuce pour générer des vraies alertes en test** : des features aléatoires dans une plage
`0–1000` retombent presque toujours sur "Benign". Utilise une plage plus large, ex.
`random.uniform(0, 100000)`, pour obtenir un taux d'alerte réaliste (~2-5%) pendant tes démos.

**⚠️ Traceback cosmétique connu** : au moment où un client WebSocket se déconnecte, le serveur de
développement Werkzeug affiche parfois `AssertionError: write() before start_response` dans le
terminal. C'est un comportement connu de Flask-SocketIO + Werkzeug en mode dev (pas de vrai serveur
de prod) — ça n'interrompt pas le serveur ni les autres clients, on peut l'ignorer en soutenance.

**Dépendance requise pour un vrai WebSocket (pas juste polling)** : `pip install simple-websocket`
(déjà dans `requirements.txt`). Sans ce package, la connexion tombe en repli "polling" — plus lente
mais fonctionnelle.

## 🐛 Bug critique trouvé et corrigé (test navigateur réel)

En testant `dashboard.html` avec un vrai navigateur headless, un bug important a été détecté :
**Chart.js et socket.io-client étaient chargés depuis un CDN externe (`cdnjs.cloudflare.com`)**.
Si ce CDN est bloqué (pare-feu d'école/entreprise, pas d'internet en salle de soutenance,
ad-blocker...), toute l'app tombe : graphiques vides, statut "déconnecté" en permanence — alors
que le backend fonctionne parfaitement.

**Correction** : les deux librairies sont maintenant **vendorisées localement** dans
`frontend/vendor/` (`socket.io.min.js`, `chart.umd.js`) et chargées via un chemin relatif au lieu
d'un CDN. Le dashboard fonctionne désormais **100% hors-ligne** — recommandé pour une démo de
soutenance où le wifi peut être capricieux. Seules les polices Google Fonts restent en ligne
(dégradation silencieuse vers une police système si indisponible, aucun impact fonctionnel).

**Vérifié avec un vrai navigateur (Playwright/Chromium) + un vrai serveur qui tourne** : login →
connexion WebSocket → graphiques peuplés avec les vraies données de la DB → capture d'écran à
l'appui.
