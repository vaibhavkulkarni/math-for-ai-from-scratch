/* On-demand contents panel.
   The sidebars are hidden for distraction-free reading. This adds a quiet header button
   (and the "t" key) that opens an overlay listing every chapter, with the current page's
   sections under the active one. Esc, a click outside, or choosing a link closes it.
   It is built from the navigation Material already renders (hidden) on every page,
   so it never goes out of date. */
(function () {
  const ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9h14V7H3zm0 4h14v-2H3zm0 4h14v-2H3zm16 0h2v-2h-2zm0-10v2h2V7zm0 6h2v-2h-2z"/></svg>';

  function text(el) {
    const span = el && el.querySelector(".md-ellipsis");
    return (span ? span.textContent : el ? el.textContent : "").trim();
  }

  function buildList() {
    const list = document.createElement("ol");
    list.className = "toc-panel__chapters";
    const items = document.querySelectorAll(".md-sidebar--primary .md-nav--primary > .md-nav__list > .md-nav__item");
    items.forEach((item) => {
      const link = item.querySelector(":scope > a.md-nav__link");
      if (!link) return;
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = link.getAttribute("href");
      a.textContent = text(link);
      li.appendChild(a);
      if (item.classList.contains("md-nav__item--active")) {
        li.className = "is-current";
        a.setAttribute("aria-current", "page");
        const sections = item.querySelectorAll(":scope > nav.md-nav--secondary > ul > li > a.md-nav__link");
        if (sections.length) {
          const sub = document.createElement("ul");
          sections.forEach((s) => {
            const sli = document.createElement("li");
            const sa = document.createElement("a");
            sa.href = s.getAttribute("href");
            sa.textContent = text(s);
            sli.appendChild(sa);
            sub.appendChild(sli);
          });
          li.appendChild(sub);
        }
      }
      list.appendChild(li);
    });
    return list;
  }

  function panel() {
    let p = document.getElementById("toc-panel");
    if (p) return p;
    p = document.createElement("div");
    p.id = "toc-panel";
    p.className = "toc-panel";
    p.hidden = true;
    p.innerHTML =
      '<div class="toc-panel__backdrop" data-close></div>' +
      '<div class="toc-panel__box" role="dialog" aria-modal="true" aria-label="Contents">' +
      '<div class="toc-panel__head"><span>Contents</span>' +
      '<button class="toc-panel__close" type="button" aria-label="Close contents" data-close>&times;</button></div>' +
      '<nav class="toc-panel__body"></nav>' +
      '<p class="toc-panel__hint">Press <kbd>t</kbd> to open or close, <kbd>Esc</kbd> to close.</p></div>';
    p.addEventListener("click", (e) => {
      if (e.target.closest("[data-close]") || e.target.closest(".toc-panel__body a")) close();
    });
    document.body.appendChild(p);
    return p;
  }

  function open() {
    const p = panel();
    const body = p.querySelector(".toc-panel__body");
    body.replaceChildren(buildList());
    p.hidden = false;
    document.documentElement.classList.add("toc-open");
    const current = body.querySelector(".is-current > a") || body.querySelector("a");
    if (current) {
      current.focus({ preventScroll: true });
      current.scrollIntoView({ block: "center" });
    }
  }

  function close() {
    const p = document.getElementById("toc-panel");
    if (!p || p.hidden) return;
    p.hidden = true;
    document.documentElement.classList.remove("toc-open");
    const btn = document.getElementById("toc-button");
    if (btn) btn.focus({ preventScroll: true });
  }

  function toggle() {
    const p = document.getElementById("toc-panel");
    p && !p.hidden ? close() : open();
  }

  function addButton() {
    if (document.getElementById("toc-button")) return;
    const logo = document.querySelector(".md-header__inner .md-logo");
    if (!logo) return;
    const btn = document.createElement("button");
    btn.id = "toc-button";
    btn.type = "button";
    btn.className = "md-header__button md-icon toc-button";
    btn.title = "Contents (t)";
    btn.setAttribute("aria-label", "Open contents");
    btn.innerHTML = ICON;
    btn.addEventListener("click", toggle);
    logo.insertAdjacentElement("afterend", btn);
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") return close();
    const t = e.target;
    const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "t") { e.preventDefault(); toggle(); }
  });

  // Material's instant navigation swaps pages without a reload; re-check on every page.
  if (typeof document$ !== "undefined") {
    document$.subscribe(() => { addButton(); close(); });
  } else {
    document.addEventListener("DOMContentLoaded", addButton);
  }
})();
