// collapsible tree view of a psin metadata file
// blocks open into pages of items, values are edited in place and collected by file offset
use std::collections::{BTreeMap, HashMap};

use eframe::egui::{self, Color32, RichText};

use crate::psin_xml::{self, Applied, BlockInfo, Node, Value};

// items shown per page of a block
const PAGE: usize = 50;
// pages kept decoded
const CACHED_PAGES: usize = 24;

struct Edit {
    ty: String,
    text: String,
    label: String,
    valid: bool,
}

pub struct PsinTree {
    data: Vec<u8>,
    blocks: Vec<BlockInfo>,
    // first item shown of every paged block
    first: HashMap<usize, usize>,
    pages: HashMap<(usize, usize), Vec<Vec<Node>>>,
    edits: BTreeMap<usize, Edit>,
    filter: String,
    // block and item to open and scroll to on the next frame
    jump: Option<(usize, usize)>,
}

// what drawing a row may ask for
#[derive(Default)]
struct Asked {
    // block and byte offset of a pointer that was clicked
    follow: Option<(usize, usize)>,
}

impl PsinTree {
    pub fn new(data: Vec<u8>) -> Result<Self, String> {
        let blocks = psin_xml::outline(&data)?;
        Ok(Self { data, blocks, first: HashMap::new(), pages: HashMap::new(), edits: BTreeMap::new(), filter: String::new(), jump: None })
    }

    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    pub fn invalid_edits(&self) -> usize {
        self.edits.values().filter(|e| !e.valid).count()
    }

    pub fn clear_edits(&mut self) {
        self.edits.clear();
    }

    // sets the text of a value the way typing into its box does
    #[cfg(test)]
    fn set(&mut self, value: &Value, text: &str) {
        if let Some(pos) = value.offset {
            set_text(&mut self.edits, pos, value, text.to_string());
        }
    }

    // a copy of the file with every edit written in
    pub fn patched(&self) -> Result<Applied, String> {
        psin_xml::patch(&self.data, self.edits.iter().map(|(&pos, e)| (pos, e.ty.as_str(), e.text.as_str(), e.label.as_str())))
    }

    #[cfg(test)]
    fn open_at(&mut self, block: usize, item: usize) {
        self.jump = Some((block, item));
    }

    fn page(&mut self, block: usize, first: usize) -> Vec<Vec<Node>> {
        if let Some(page) = self.pages.remove(&(block, first)) {
            return page;
        }
        if self.pages.len() >= CACHED_PAGES {
            self.pages.clear();
        }
        psin_xml::items(&self.data, block, first, PAGE).unwrap_or_default()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Filter blocks by type").desired_width(180.0));
            let items: usize = self.blocks.iter().map(|b| b.count).sum();
            ui.label(RichText::new(format!("{} blocks · {} items", self.blocks.len(), items)).size(11.0).color(Color32::from_rgb(150, 160, 180)));
        });
        let jump = self.jump.take();
        let filter = self.filter.to_lowercase();
        let mut asked = Asked::default();
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            for bi in 0..self.blocks.len() {
                let (count, size) = (self.blocks[bi].count, self.blocks[bi].size);
                let jump_here = jump.filter(|j| j.0 == bi).map(|j| j.1);
                if jump_here.is_none() && !filter.is_empty() && !self.blocks[bi].label.to_lowercase().contains(&filter) {
                    continue;
                }
                let detail = if count == 1 { format!("{size} bytes") } else { format!("{count} items") };
                let title = RichText::new(format!("{bi} · {} · {detail}", self.blocks[bi].label)).monospace();
                let mut header = egui::CollapsingHeader::new(title).id_salt(("psin block", bi));
                if let Some(item) = jump_here {
                    header = header.open(Some(true));
                    self.first.insert(bi, item / PAGE * PAGE);
                }
                let shown = header.show(ui, |ui| self.block_ui(ui, bi, jump_here, &mut asked));
                // an item that was asked for scrolls itself into view
                if jump_here.is_some() && shown.body_returned != Some(true) {
                    shown.header_response.scroll_to_me(Some(egui::Align::TOP));
                }
            }
        });
        if let Some((block, offset)) = asked.follow {
            let item = self.blocks.get(block).map_or(0, |b| if b.item_size == 0 { 0 } else { offset / b.item_size });
            self.jump = Some((block, item));
            ui.ctx().request_repaint();
        }
    }

    // true when the item asked for was scrolled to
    fn block_ui(&mut self, ui: &mut egui::Ui, bi: usize, jump: Option<usize>, asked: &mut Asked) -> bool {
        let count = self.blocks[bi].count;
        let mut first = self.first.get(&bi).copied().unwrap_or(0).min(count.saturating_sub(1) / PAGE * PAGE);
        if count > PAGE {
            ui.horizontal(|ui| {
                if ui.add_enabled(first > 0, egui::Button::new("Previous")).clicked() {
                    first -= PAGE;
                }
                if ui.add_enabled(first + PAGE < count, egui::Button::new("Next")).clicked() {
                    first += PAGE;
                }
                ui.label(format!("items {first} to {} of {count}", (first + PAGE).min(count) - 1));
                let mut go = first;
                if ui.add(egui::DragValue::new(&mut go).range(0..=count - 1).prefix("go to ")).changed() {
                    first = go / PAGE * PAGE;
                }
            });
            self.first.insert(bi, first);
        }
        let page = self.page(bi, first);
        // one struct of its own needs no item level, plain values are one row each
        let flat = count == 1 || page.iter().all(|nodes| matches!(nodes.as_slice(), [Node::Value(_)]));
        let mut scrolled = false;
        for (k, nodes) in page.iter().enumerate() {
            let item = first + k;
            ui.push_id(item, |ui| {
                if flat {
                    nodes_ui(ui, nodes, &mut self.edits, asked);
                    return;
                }
                let mut header = egui::CollapsingHeader::new(RichText::new(item_title(item, nodes)).monospace()).id_salt("item");
                if jump == Some(item) {
                    header = header.open(Some(true));
                }
                let shown = header.show(ui, |ui| nodes_ui(ui, nodes, &mut self.edits, asked));
                if jump == Some(item) {
                    shown.header_response.scroll_to_me(Some(egui::Align::TOP));
                    scrolled = true;
                }
            });
        }
        self.pages.insert((bi, first), page);
        scrolled
    }
}

// an item is told apart by the first name or text it holds
fn item_title(index: usize, nodes: &[Node]) -> String {
    let hint = nodes.iter().find_map(|n| match n {
        Node::Value(v) if v.ty == "hash" && !v.note.is_empty() && v.note != "null" => Some(v.note.clone()),
        Node::Value(v) if v.ty == "string" && !v.text.is_empty() => Some(v.text.clone()),
        _ => None,
    });
    match hint {
        Some(h) => format!("[{index}] {h}"),
        None => format!("[{index}]"),
    }
}

fn nodes_ui(ui: &mut egui::Ui, nodes: &[Node], edits: &mut BTreeMap<usize, Edit>, asked: &mut Asked) {
    for (i, node) in nodes.iter().enumerate() {
        ui.push_id(i, |ui| match node {
            Node::Value(v) => value_ui(ui, v, edits, asked),
            Node::Group { label, kind, children } => {
                egui::CollapsingHeader::new(RichText::new(format!("{label}  {kind}")).monospace())
                    .id_salt("group")
                    .default_open(children.len() <= 12)
                    .show(ui, |ui| nodes_ui(ui, children, edits, asked));
            }
        });
    }
}

// records the text typed for a value, text equal to the file's own is no edit
fn set_text(edits: &mut BTreeMap<usize, Edit>, pos: usize, v: &Value, text: String) {
    if text == v.text {
        edits.remove(&pos);
    } else {
        let valid = psin_xml::check(&v.ty, &text).is_ok();
        edits.insert(pos, Edit { ty: v.ty.clone(), text, label: v.label.clone(), valid });
    }
}

fn value_ui(ui: &mut egui::Ui, v: &Value, edits: &mut BTreeMap<usize, Edit>, asked: &mut Asked) {
    let dim = Color32::from_rgb(150, 160, 180);
    ui.horizontal(|ui| {
        ui.label(RichText::new(&v.label).monospace());
        ui.label(RichText::new(&v.ty).size(11.0).color(dim));
        if let Some(pos) = v.offset {
            let edit = edits.get(&pos);
            let mut text = edit.map_or_else(|| v.text.clone(), |e| e.text.clone());
            let colour = match edit {
                Some(e) if !e.valid => Some(Color32::from_rgb(235, 100, 100)),
                Some(_) => Some(Color32::from_rgb(240, 190, 90)),
                None => None,
            };
            let width = (text.chars().count().max(6) as f32 * 7.5 + 14.0).min(460.0);
            let mut field = egui::TextEdit::singleline(&mut text).font(egui::TextStyle::Monospace).desired_width(width);
            if let Some(c) = colour {
                field = field.text_color(c);
            }
            let shown = ui.add(field);
            if shown.changed() {
                set_text(edits, pos, v, text);
            }
            if let Some(e) = edits.get(&pos) {
                if !e.valid {
                    ui.label(RichText::new(format!("not a valid {}", v.ty)).size(11.0).color(Color32::from_rgb(235, 100, 100)));
                }
                if ui.small_button("Reset").on_hover_text(format!("Back to {}", v.text)).clicked() {
                    edits.remove(&pos);
                }
            }
        } else if !v.text.is_empty() {
            ui.label(RichText::new(&v.text).monospace());
        }
        if let Some((block, offset)) = v.to {
            let text = if offset == 0 { format!("block {block}") } else { format!("block {block} +0x{offset:x}") };
            if ui.link(text).on_hover_text("Open the block this points to").clicked() {
                asked.follow = Some((block, offset));
            }
        }
        if !v.note.is_empty() {
            ui.label(RichText::new(&v.note).color(dim));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names;

    fn drawn(shape: &egui::Shape, out: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(t) => out.push(t.galley.text().to_string()),
            egui::Shape::Vec(v) => v.iter().for_each(|s| drawn(s, out)),
            _ => {}
        }
    }

    // text on screen after a few frames, enough for opened headers to lay out their content
    fn render(tree: &mut PsinTree) -> Vec<String> {
        let ctx = egui::Context::default();
        let mut texts = Vec::new();
        for _ in 0..4 {
            let out = ctx.run(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 2000.0))), ..Default::default() }, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| tree.ui(ui));
            });
            texts.clear();
            out.shapes.iter().for_each(|s| drawn(&s.shape, &mut texts));
        }
        texts
    }

    fn value<'a>(nodes: &'a [Node], label: &str) -> &'a Value {
        nodes
            .iter()
            .find_map(|n| match n {
                Node::Value(v) if v.label == label => Some(v),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{label} missing"))
    }

    #[test]
    fn blocks_open_into_rows_and_edits_patch_the_file() {
        names::add(["TestThing", "Mass", "Gears", "Model", "Enabled", "Ratio", "Offset", "Curve", "cheetah2"]);
        let file = psin_xml::tests::sample();
        let mut tree = PsinTree::new(file.clone()).expect("tree");
        let closed = render(&mut tree);
        assert!(closed.iter().any(|t| t == "0 · TestThing · 64 bytes"), "{closed:?}");
        assert!(closed.iter().any(|t| t == "1 · f32 · 2 items"), "{closed:?}");
        assert!(!closed.iter().any(|t| t == "Mass"));

        tree.open_at(0, 0);
        let open = render(&mut tree);
        for want in ["Mass", "1630", "Gears", "-7", "cheetah2", "Curve", "2 values", "block 1"] {
            assert!(open.iter().any(|t| t == want), "{want} missing in {open:?}");
        }

        let nodes = psin_xml::items(&file, 0, 0, PAGE).expect("items").remove(0);
        let mass = value(&nodes, "Mass").clone();
        tree.set(&mass, "heavy");
        assert_eq!((tree.edit_count(), tree.invalid_edits()), (1, 1));
        tree.open_at(0, 0);
        assert!(render(&mut tree).iter().any(|t| t == "not a valid f32"));
        assert!(tree.patched().is_err());
        tree.set(&mass, "1800.5");
        tree.set(value(&nodes, "Gears"), "-7");
        assert_eq!((tree.edit_count(), tree.invalid_edits()), (1, 0));
        let done = tree.patched().expect("patch");
        assert_eq!(done.changes.len(), 1, "{:?}", done.changes);
        let xml = psin_xml::export_xml(&file, 0).unwrap().replace("v=\"1630\"", "v=\"1800.5\"");
        assert_eq!(done.data, psin_xml::apply_xml(&file, &xml).unwrap().data);
        tree.set(&mass, "1630");
        assert_eq!(tree.edit_count(), 0);
    }

    fn editable(nodes: &[Node]) -> usize {
        nodes
            .iter()
            .map(|n| match n {
                Node::Value(v) => v.offset.is_some() as usize,
                Node::Group { children, .. } => editable(children),
            })
            .sum()
    }

    // the tree of every extracted file holds exactly the editable values its xml does, PERCYSTAR_PSIN_DIR names the folder
    #[test]
    fn extracted_files_match_their_xml() {
        let Ok(dir) = std::env::var("PERCYSTAR_PSIN_DIR") else { return };
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let (mut files, mut values) = (0usize, 0usize);
        for entry in rd.flatten() {
            let Ok(file) = std::fs::read(entry.path()) else { continue };
            let Ok(xml) = psin_xml::export_xml(&file, 0) else { continue };
            let blocks = psin_xml::outline(&file).expect("outline");
            let mut seen = 0usize;
            for (bi, b) in blocks.iter().enumerate() {
                let mut first = 0;
                while first < b.count {
                    let page = psin_xml::items(&file, bi, first, 4096).unwrap_or_else(|e| panic!("{}: {e}", entry.path().display()));
                    assert!(!page.is_empty(), "{}: block {bi} ends early at {first} of {}", entry.path().display(), b.count);
                    seen += page.iter().map(|n| editable(n)).sum::<usize>();
                    first += page.len();
                }
            }
            assert_eq!(seen, xml.matches(" o=\"0x").count(), "{}", entry.path().display());
            files += 1;
            values += seen;
        }
        eprintln!("{files} files: tree and xml agree on {values} editable values");
    }
}
