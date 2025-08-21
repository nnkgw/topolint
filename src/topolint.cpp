// Non-manifold detector + visualizer (freeglut + glm)
// Detects and colorizes:
// 1) Y-edge (edge incident to >2 faces)        -> RED lines
// 2) Non-manifold vertex (broken fan, etc.)    -> YELLOW points
// 3) T-junction (vertex lies on interior edge) -> CYAN points + edges
#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/glut.h>
#include <GL/freeglut.h>
#ifdef NDEBUG
//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#endif // NDEBUG

#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else // MACOSX
#include <GL/glut.h>
#include <GL/freeglut.h>
#endif // unix

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <queue>
#include <tuple>
#include <algorithm>
#include <cstring>

struct Face { int a, b, c; }; // 0-based indices

struct EdgeKey {
  int u, v; // u < v
  EdgeKey() : u(0), v(0) {}
  EdgeKey(int a, int b) { if (a < b) { u = a; v = b; } else { u = b; v = a; } }
  bool operator==(const EdgeKey& o) const { return u == o.u && v == o.v; }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey& e) const {
    return (static_cast<size_t>(e.u) << 32) | static_cast<size_t>(e.v);
  }
};

static bool starts_with(const std::string& s, const char* p) {
  return s.size() >= std::strlen(p) && std::equal(p, p + std::strlen(p), s.begin());
}

// Parse integer before first '/' (OBJ index)
static int parse_index_token(const std::string& tok) {
  int i = 0;
  size_t pos = 0;
  while (pos < tok.size() && tok[pos] != '/') pos++;
  std::string head = tok.substr(0, pos);
  if (head.empty()) return 0;
  try { i = std::stoi(head); } catch (...) { i = 0; }
  return i;
}

// Convert possibly negative OBJ index (1-based, negatives are relative)
static int resolve_index(int idx, int n) {
  if (idx > 0) return idx - 1;
  if (idx < 0) return n + idx;
  return -1;
}

struct Mesh {
  std::vector<glm::vec3> V;
  std::vector<Face> F;
};

// ---------- OBJ loader ----------
static bool load_obj(const std::string& path, Mesh& mesh) {
  std::ifstream ifs(path);
  if (!ifs) { std::cerr << "Cannot open: " << path << "\n"; return false; }
  std::string line;
  std::vector<glm::vec3> V;
  std::vector<Face> F;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    if (starts_with(line, "v ")) {
      std::istringstream iss(line);
      char ch; float x, y, z;
      iss >> ch >> x >> y >> z;
      V.emplace_back(x, y, z);
    } else if (starts_with(line, "f ")) {
      std::istringstream iss(line);
      char ch; iss >> ch;
      std::vector<int> idxs; std::string tok;
      while (iss >> tok) {
        int raw = parse_index_token(tok);
        int id = resolve_index(raw, (int)V.size());
        if (id < 0 || id >= (int)V.size()) {
          std::cerr << "Invalid index in face: " << tok << "\n"; return false;
        }
        idxs.push_back(id);
      }
      if (idxs.size() < 3) continue;
      for (size_t k = 1; k + 1 < idxs.size(); ++k) F.push_back({idxs[0], idxs[k], idxs[k + 1]});
    }
  }
  mesh.V.swap(V); mesh.F.swap(F);
  return true;
}

// ---------- utilities ----------
static double bbox_diag(const std::vector<glm::vec3>& V, glm::vec3* out_lo=nullptr, glm::vec3* out_hi=nullptr) {
  glm::vec3 lo( std::numeric_limits<float>::infinity());
  glm::vec3 hi(-std::numeric_limits<float>::infinity());
  for (auto& p : V) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
  if (out_lo) *out_lo = lo; if (out_hi) *out_hi = hi;
  return glm::length(hi - lo);
}

static double point_segment_distance(const glm::vec3& A, const glm::vec3& B, const glm::vec3& P) {
  glm::vec3 AB = B - A;
  float ab2 = glm::dot(AB, AB);
  if (ab2 == 0.0f) return glm::length(P - A);
  float t = glm::dot(P - A, AB) / ab2;
  t = std::max(0.0f, std::min(1.0f, t));
  glm::vec3 Q = A + t * AB;
  return glm::length(P - Q);
}

static void build_edge_faces(
  const std::vector<Face>& F,
  std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>& edge2faces) {
  for (int fi = 0; fi < (int)F.size(); ++fi) {
    int a = F[fi].a, b = F[fi].b, c = F[fi].c;
    edge2faces[EdgeKey(a, b)].push_back(fi);
    edge2faces[EdgeKey(b, c)].push_back(fi);
    edge2faces[EdgeKey(c, a)].push_back(fi);
  }
}

static void build_vertex_faces(
  int nv, const std::vector<Face>& F,
  std::vector<std::vector<int>>& vfaces) {
  vfaces.assign(nv, {});
  for (int fi = 0; fi < (int)F.size(); ++fi) {
    vfaces[F[fi].a].push_back(fi);
    vfaces[F[fi].b].push_back(fi);
    vfaces[F[fi].c].push_back(fi);
  }
}

static bool vertex_is_manifold(
  int v,
  const std::vector<Face>& F,
  const std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash>& edge2faces,
  const std::vector<std::vector<int>>& vfaces) {
  const auto& faces = vfaces[v];
  if (faces.empty()) return true;

  std::unordered_map<int, std::vector<int>> adj; // face -> neighbors via edges touching v
  int boundary_edge_count = 0;
  for (int fi : faces) {
    const Face& T = F[fi];
    int vs[3] = {T.a, T.b, T.c};
    for (int k = 0; k < 3; ++k) {
      int x = vs[k], y = vs[(k + 1) % 3];
      if (x != v && y != v) continue;
      int other = (x == v) ? y : x;
      const auto& inc = edge2faces.at(EdgeKey(v, other));
      if (inc.size() == 1) boundary_edge_count++; // boundary at v
      for (int fj : inc) if (fj != fi) adj[fi].push_back(fj);
    }
  }

  std::unordered_set<int> seen;
  int comp = 0;
  for (int fi : faces) {
    if (seen.count(fi)) continue;
    comp++;
    std::queue<int> q; q.push(fi); seen.insert(fi);
    while (!q.empty()) {
      int cur = q.front(); q.pop();
      for (int nb : adj[cur]) if (!seen.count(nb)) { seen.insert(nb); q.push(nb); }
    }
  }
  if (comp != 1) return false;
  if (!(boundary_edge_count == 0 || boundary_edge_count == 2)) return false;
  return true;
}

// ---------- global state (mesh + detections) ----------
static Mesh g_mesh;
static std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> g_edge2faces;
static std::vector<std::vector<int>> g_vfaces;

static std::vector<EdgeKey> g_y_edges;              // red
static std::vector<int> g_nm_vertices;              // yellow
struct TJ { int v; EdgeKey e; };
static std::vector<TJ> g_tjs;                       // cyan
static std::unordered_set<int> g_tj_vertices;       // cyan points
static std::unordered_set<EdgeKey, EdgeKeyHash> g_tj_edges; // cyan lines

// view state
static glm::vec3 g_center(0.0f);
static float g_radius = 1.0f;
static float g_dist = 3.0f, g_yaw = 0.0f, g_pitch = 0.0f;
static int g_lastx = 0, g_lasty = 0;
static bool g_drag_rot=false, g_drag_zoom=false, g_drag_pan=false;
static float g_panx=0.0f, g_pany=0.0f;

// toggles
static bool g_show_yedge = true;
static bool g_show_nmvert = true;
static bool g_show_tj = true;

// ---------- detection ----------
static void detect_nonmanifold() {
  g_y_edges.clear(); g_nm_vertices.clear();
  g_tjs.clear(); g_tj_vertices.clear(); g_tj_edges.clear();

  // Y-edge
  for (const auto& kv : g_edge2faces) if (kv.second.size() > 2) g_y_edges.push_back(kv.first);

  // Non-manifold vertex
  for (int v = 0; v < (int)g_mesh.V.size(); ++v)
    if (!vertex_is_manifold(v, g_mesh.F, g_edge2faces, g_vfaces)) g_nm_vertices.push_back(v);

  // T-junctions
  double diag = bbox_diag(g_mesh.V);
  double eps = std::max(1e-7, 1e-6 * diag);

  for (const auto& kv : g_edge2faces) {
    EdgeKey e = kv.first;
    const glm::vec3& A = g_mesh.V[e.u];
    const glm::vec3& B = g_mesh.V[e.v];
    double ab_len = glm::length(B - A);
    if (ab_len <= eps) continue;

    for (int vi = 0; vi < (int)g_mesh.V.size(); ++vi) {
      if (vi == e.u || vi == e.v) continue;

      bool shares_edge = false;
      for (int fi : g_vfaces[vi]) {
        const Face& T = g_mesh.F[fi];
        if ((EdgeKey(T.a, T.b) == e) || (EdgeKey(T.b, T.c) == e) || (EdgeKey(T.c, T.a) == e)) {
          shares_edge = true; break;
        }
      }
      if (shares_edge) continue;

      double d = point_segment_distance(A, B, g_mesh.V[vi]);
      if (d <= eps) {
        glm::vec3 AB = B - A;
        float t = glm::dot(g_mesh.V[vi] - A, AB) / glm::dot(AB, AB);
        if (t > 1e-6 && t < 1.0f - 1e-6) {
          g_tjs.push_back({vi, e});
          g_tj_vertices.insert(vi);
          g_tj_edges.insert(e);
        }
      }
    }
  }

  // Print summary
  std::cout << "Y-edges: " << g_y_edges.size() << "\n";
  std::cout << "Non-manifold vertices: " << g_nm_vertices.size() << "\n";
  std::cout << "T-junction vertices: " << g_tj_vertices.size()
            << "  (edges: " << g_tj_edges.size() << ")\n";
}

// ---------- drawing ----------
static void draw_string(float x, float y, const char* s) {
  glRasterPos2f(x, y);
  for (const char* p = s; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *p);
}

static void apply_camera() {
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float cp = cosf(g_pitch), sp = sinf(g_pitch);
  float cy = cosf(g_yaw),   sy = sinf(g_yaw);
  glm::vec3 eye(
    g_center.x + g_dist * cp * sy + g_panx,
    g_center.y + g_dist * sp      + g_pany,
    g_center.z + g_dist * cp * cy
  );
  gluLookAt(eye.x, eye.y, eye.z, g_center.x + g_panx, g_center.y + g_pany, g_center.z, 0, 1, 0);
}

static void draw_mesh_wire() {
  glColor3f(0.75f, 0.75f, 0.75f);
  glLineWidth(1.0f);
  glBegin(GL_LINES);
  for (const Face& f : g_mesh.F) {
    const glm::vec3& a = g_mesh.V[f.a];
    const glm::vec3& b = g_mesh.V[f.b];
    const glm::vec3& c = g_mesh.V[f.c];
    glVertex3f(a.x, a.y, a.z); glVertex3f(b.x, b.y, b.z);
    glVertex3f(b.x, b.y, b.z); glVertex3f(c.x, c.y, c.z);
    glVertex3f(c.x, c.y, c.z); glVertex3f(a.x, a.y, a.z);
  }
  glEnd();
}

static void draw_y_edges() {
  if (!g_show_yedge) return;
  glColor3f(1.0f, 0.2f, 0.2f); // red
  glLineWidth(4.0f);
  glBegin(GL_LINES);
  for (const auto& e : g_y_edges) {
    const glm::vec3& a = g_mesh.V[e.u];
    const glm::vec3& b = g_mesh.V[e.v];
    glVertex3f(a.x, a.y, a.z); glVertex3f(b.x, b.y, b.z);
  }
  glEnd();
}

static void draw_nm_vertices() {
  if (!g_show_nmvert) return;
  glColor3f(1.0f, 1.0f, 0.0f); // yellow
  glPointSize(10.0f);
  glBegin(GL_POINTS);
  for (int v : g_nm_vertices) {
    const auto& p = g_mesh.V[v];
    glVertex3f(p.x, p.y, p.z);
  }
  glEnd();
}

static void draw_tjunctions() {
  if (!g_show_tj) return;
  // edges
  glColor3f(0.0f, 1.0f, 1.0f); // cyan
  glLineWidth(3.0f);
  glBegin(GL_LINES);
  for (const auto& e : g_tj_edges) {
    const glm::vec3& a = g_mesh.V[e.u];
    const glm::vec3& b = g_mesh.V[e.v];
    glVertex3f(a.x, a.y, a.z); glVertex3f(b.x, b.y, b.z);
  }
  glEnd();
  // points
  glPointSize(9.0f);
  glBegin(GL_POINTS);
  for (int v : g_tj_vertices) {
    const auto& p = g_mesh.V[v];
    glVertex3f(p.x, p.y, p.z);
  }
  glEnd();
}

static void draw_overlay() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix(); glLoadIdentity();
  glDisable(GL_DEPTH_TEST);

  draw_string(0.02f, 0.96f, "LMB: rotate, MMB: pan, RMB: zoom  |  1:Y-edge  2:NM-vertex  3:T-junction  r:reset  q:quit");
  glColor3f(1,0.2f,0.2f); draw_string(0.02f, 0.92f, "Y-edge: RED");
  glColor3f(1,1,0);      draw_string(0.20f, 0.92f, "Non-manifold vertex: YELLOW");
  glColor3f(0,1,1);      draw_string(0.52f, 0.92f, "T-junction: CYAN");

  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

static void display_cb() {
  glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, glutGet(GLUT_WINDOW_WIDTH) / (double)glutGet(GLUT_WINDOW_HEIGHT), g_radius * 0.01, g_radius * 100.0);

  apply_camera();
  draw_mesh_wire();
  draw_y_edges();
  draw_nm_vertices();
  draw_tjunctions();
  draw_overlay();

  glutSwapBuffers();
}

static void reshape_cb(int w, int h) {
  glViewport(0, 0, w, h > 0 ? h : 1);
}

static void keyboard_cb(unsigned char key, int, int) {
  if (key == 'q' || key == 27) std::exit(0);
  if (key == 'r') { g_yaw = g_pitch = 0.0f; g_dist = 3.0f * g_radius; g_panx = g_pany = 0.0f; }
  if (key == '1') g_show_yedge = !g_show_yedge;
  if (key == '2') g_show_nmvert = !g_show_nmvert;
  if (key == '3') g_show_tj = !g_show_tj;
  glutPostRedisplay();
}

static void mouse_cb(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    g_lastx = x; g_lasty = y;
    if (button == GLUT_LEFT_BUTTON)  g_drag_rot  = true;
    if (button == GLUT_RIGHT_BUTTON) g_drag_zoom = true;
    if (button == GLUT_MIDDLE_BUTTON) g_drag_pan = true;
  } else {
    g_drag_rot = g_drag_zoom = g_drag_pan = false;
  }
}

static void motion_cb(int x, int y) {
  int dx = x - g_lastx, dy = y - g_lasty;
  g_lastx = x; g_lasty = y;
  if (g_drag_rot) {
    g_yaw   += dx * 0.01f;
    g_pitch += dy * 0.01f;
    g_pitch = std::max(-1.5f, std::min(1.5f, g_pitch));
  }
  if (g_drag_zoom) {
    g_dist *= (1.0f + dy * 0.01f);
    g_dist = std::max(0.1f * g_radius, std::min(50.0f * g_radius, g_dist));
  }
  if (g_drag_pan) {
    float s = 0.0025f * g_dist;
    g_panx -= dx * s;
    g_pany += dy * s;
  }
  glutPostRedisplay();
}

// ---------- main ----------
int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "Usage: " << argv[0] << " mesh.obj\n"; return 1; }

  glutInit(&argc, argv); // real GLUT init (we will open a window)

  if (!load_obj(argv[1], g_mesh)) return 1;
  std::cout << "Loaded V=" << g_mesh.V.size() << " F=" << g_mesh.F.size() << "\n";

  build_edge_faces(g_mesh.F, g_edge2faces);
  build_vertex_faces((int)g_mesh.V.size(), g_mesh.F, g_vfaces);
  detect_nonmanifold();

  glm::vec3 lo, hi;
  bbox_diag(g_mesh.V, &lo, &hi);
  g_center = 0.5f * (lo + hi);
  g_radius = std::max(1e-4f, 0.5f * glm::length(hi - lo));
  g_dist   = 3.0f * g_radius;

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1200, 800);
  glutCreateWindow("Non-manifold Visualizer (Y-edge=RED, NM-vertex=YELLOW, T-junction=CYAN)");

  glutDisplayFunc(display_cb);
  glutReshapeFunc(reshape_cb);
  glutKeyboardFunc(keyboard_cb);
  glutMouseFunc(mouse_cb);
  glutMotionFunc(motion_cb);

  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glPointSize(8.0f);

  glutMainLoop();
  return 0;
}
