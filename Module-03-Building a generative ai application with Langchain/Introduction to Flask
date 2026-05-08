# Introduction to Flask

> A concise reference on the Flask web framework — its features, dependencies, extensions, installation, and how it compares to Django.

---

## Table of Contents

1. [What is Flask?](#what-is-flask)
2. [Core Features](#core-features)
3. [Additional Features](#additional-features)
4. [Community Extensions](#community-extensions)
5. [Installation](#installation)
6. [Built-in Dependencies](#built-in-dependencies)
7. [Flask vs Django](#flask-vs-django)
8. [Summary](#summary)

---

## What is Flask?

Flask is a **micro web framework** for Python used to build web applications. Key characteristics:

- **Not opinionated** — does not bind you to specific tools or patterns
- **Minimal by default** — ships with only the essentials
- **Highly extensible** — community extensions add any additional functionality you need
- Requires a minimum of **Python 3.7** (for Flask 2.2.2)

> Flask was created in **2004 by Armin Ronacher** — originally as an April Fool's joke — and quickly gained widespread popularity for its ease of use and extensibility.

---

## Core Features

| Feature | Description |
|---|---|
| **Development Web Server** | Built-in server to run and test applications locally |
| **Debugger** | Interactive traceback and stack trace displayed in the browser |
| **Logging** | Uses standard Python logging; supports custom log messages |
| **Testing Support** | Built-in test client; integrates with frameworks like `pytest` and `coverage` for test-driven development |
| **Request & Response Objects** | Access and manipulate HTTP request arguments and customize responses |

---

## Additional Features

| Feature | Description |
|---|---|
| **Static Assets** | Supports CSS, JavaScript, and image files via template tags |
| **Jinja Templating** | Dynamic pages using Jinja2 — display changing data, check login state, etc. |
| **Routing & Dynamic URLs** | Define routes for different HTTP methods; supports RESTful services and URL redirection |
| **Error Handlers** | Write global, application-level error handlers |
| **Session Management** | Built-in support for managing user sessions |

---

## Community Extensions

Flask's plugin ecosystem lets you add functionality in a plug-and-play manner.

### Database & ORM

| Extension | Purpose |
|---|---|
| **Flask-SQLAlchemy** | Adds SQLAlchemy ORM support — work with databases using Python objects |
| **Flask-Migrate** | Adds database migration support on top of SQLAlchemy |

### User Management

| Extension | Purpose |
|---|---|
| **Flask-User** | User authentication, authorization, and user management |
| **Flask-Admin** | Easily add admin interfaces to Flask applications |

### Communication & Files

| Extension | Purpose |
|---|---|
| **Flask-Mail** | Configure and send emails via an SMTP mail server |
| **Flask-Uploads** | Customized file upload handling |

### Data & Tasks

| Extension | Purpose |
|---|---|
| **Marshmallow** | Extensive object serialization and deserialization |
| **Celery** | Powerful task queue for background tasks, scheduling, and complex workflows |
| **Flask-CORS** | Handle Cross-Origin Resource Sharing (CORS) for cross-origin JavaScript requests |

---

## Installation

Flask is available via **pip**, Python's package manager.

```bash
# Step 1: Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Step 2: Install Flask with a pinned version
pip install Flask==2.2.2
```

> **Why pin the version number?**
> - Ensures the application can be **reproduced consistently** across development, staging, and production environments
> - Prevents unexpected bugs from being introduced when packages auto-update

```bash
# View all installed dependencies in the virtual environment
pip freeze
```

---

## Built-in Dependencies

When you install Flask, the following dependencies are automatically included:

| Dependency | Role |
|---|---|
| **Werkzeug** | Implements WSGI (Web Server Gateway Interface) — the standard Python interface between applications and servers |
| **Jinja2** | Template language for rendering dynamic HTML pages |
| **MarkupSafe** | Comes with Jinja2; escapes untrusted input in templates to prevent injection attacks |
| **ItsDangerous** | Securely signs data; detects tampering; protects the Flask session cookie |
| **Click** | Framework for writing command-line applications; provides the `flask` CLI command and supports custom management commands |

---

## Flask vs Django

| | Flask | Django |
|---|---|---|
| **Type** | Micro framework | Full-stack framework |
| **Dependencies** | Minimal — only the basics | Batteries-included — everything built-in |
| **Flexibility** | Very flexible — plug-and-play components | Opinionated — most decisions made for you |
| **Control** | Developer chooses tools and structure | Framework enforces conventions |
| **Best for** | APIs, microservices, lightweight apps | Large, full-featured web applications |
| **Learning curve** | Lower | Steeper |

> **Flask** gives you freedom and flexibility. **Django** gives you structure and speed for full-featured applications. Choose based on project scale and team preference.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Flask** | Lightweight, non-opinionated Python micro framework |
| **Core features** | Dev server, debugger, logging, testing, request/response objects |
| **Additional features** | Routing, Jinja templates, static assets, sessions, error handling |
| **Extensions** | Community plugins for databases, auth, email, file uploads, tasks, and more |
| **Installation** | Use `pip` inside a virtual environment; always pin version numbers |
| **Built-in deps** | Werkzeug, Jinja2, MarkupSafe, ItsDangerous, Click |
| **vs Django** | Flask = flexible & minimal; Django = full-stack & opinionated |

---

*Notes based on: Introduction to Flask*
