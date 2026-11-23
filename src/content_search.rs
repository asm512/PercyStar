// search inside the readable files of the open archives, run on all cores off the ui thread
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::tex::{decode_0drpz_text, decode_rbf0};
use crate::{read_entry_bytes, RpfEntry};

// entry kinds whose contents can be shown as text
pub const KINDS: [&str; 5] = ["GXT", "XML", "RBF0", "DAT", "PSIN"];

// longest matching line kept for the result list
const SNIPPET_CHARS: usize = 200;

#[derive(Default)]
pub struct Hits {
    // entry index and the first line holding the first search word
    pub found: HashMap<usize, String>,
    pub scanned: usize,
    pub total: usize,
    pub running: bool,
    // bumps whenever found or running changes
    pub gen: u64,
    job: u64,
}

#[derive(Default)]
pub struct ContentSearch {
    hits: Arc<Mutex<Hits>>,
}

impl ContentSearch {
    pub fn hits(&self) -> MutexGuard<'_, Hits> {
        self.hits.lock().unwrap_or_else(|p| p.into_inner())
    }

    // drops the results and stops a running search
    pub fn cancel(&self) {
        let mut h = self.hits();
        h.job += 1;
        h.gen += 1;
        h.found.clear();
        h.scanned = 0;
        h.total = 0;
        h.running = false;
    }

    // searches the given (index, entry) pairs, replacing any earlier search
    pub fn start(&self, query: &str, entries: Vec<(usize, RpfEntry)>) {
        let (inc, exc) = split_terms(query);
        if inc.is_empty() {
            self.cancel();
            return;
        }
        let job = {
            let mut h = self.hits();
            h.job += 1;
            h.gen += 1;
            h.found.clear();
            h.scanned = 0;
            h.total = entries.len();
            h.running = true;
            h.job
        };
        let hits = Arc::clone(&self.hits);
        std::thread::spawn(move || {
            let next = AtomicUsize::new(0);
            let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
            std::thread::scope(|s| {
                for _ in 0..cores {
                    s.spawn(|| loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= entries.len() || hits.lock().map_or(true, |h| h.job != job) {
                            return;
                        }
                        let (idx, e) = &entries[k];
                        let line = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scan_entry(e, &inc, &exc))).ok().flatten();
                        let Ok(mut h) = hits.lock() else { return };
                        if h.job != job {
                            return;
                        }
                        h.scanned += 1;
                        if let Some(line) = line {
                            h.found.insert(*idx, line);
                            h.gen += 1;
                        }
                    });
                }
            });
            if let Ok(mut h) = hits.lock() {
                if h.job == job {
                    h.running = false;
                    h.gen += 1;
                }
            }
        });
    }
}

// lowercased words that must all appear and -words that must not
pub fn split_terms(query: &str) -> (Vec<String>, Vec<String>) {
    let mut inc = Vec::new();
    let mut exc = Vec::new();
    for t in query.to_lowercase().split_whitespace() {
        match t.strip_prefix('-') {
            Some("") => {}
            Some(rest) => exc.push(rest.to_string()),
            None => inc.push(t.to_string()),
        }
    }
    (inc, exc)
}

// readable text of an entry of one of the searchable kinds
pub fn entry_text(kind: &str, data: &[u8]) -> Option<String> {
    match kind {
        "GXT" => decode_0drpz_text(data),
        "XML" => Some(String::from_utf8_lossy(data).into_owned()),
        "RBF0" => decode_rbf0(data),
        "DAT" => Some(crate::dat::parse_dat(data)),
        "PSIN" => crate::psin_xml::export_xml(data, crate::PSIN_PREVIEW_CHARS).ok(),
        _ => None,
    }
}

// first line holding the first word when the text matches the whole query
pub fn match_text(text: &str, inc: &[String], exc: &[String]) -> Option<String> {
    let low = text.to_lowercase();
    if !inc.iter().all(|t| low.contains(t.as_str())) || exc.iter().any(|t| low.contains(t.as_str())) {
        return None;
    }
    let first = inc.first()?;
    let line = text.lines().find(|l| l.to_lowercase().contains(first.as_str())).unwrap_or_default().trim();
    Some(line.chars().take(SNIPPET_CHARS).collect())
}

fn scan_entry(e: &RpfEntry, inc: &[String], exc: &[String]) -> Option<String> {
    let data = read_entry_bytes(e).ok()?;
    match_text(&entry_text(e.kind, &data)?, inc, exc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_and_matching() {
        let (inc, exc) = split_terms("Mission  -debug -");
        assert_eq!(inc, ["mission"]);
        assert_eq!(exc, ["debug"]);
        let text = "HEADER\nStart the MISSION now\nother";
        assert_eq!(match_text(text, &inc, &exc).as_deref(), Some("Start the MISSION now"));
        assert_eq!(match_text("mission debug", &inc, &exc), None);
        let (inc2, _) = split_terms("start other");
        assert!(match_text(text, &inc2, &[]).is_some());
        let (inc3, _) = split_terms("start missing");
        assert!(match_text(text, &inc3, &[]).is_none());
    }
}
