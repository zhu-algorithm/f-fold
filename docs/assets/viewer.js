/* Wnt-pathway structure viewer for the F-Fold project page.
   Coordinates are experimental crystal structures from the RCSB PDB, trimmed
   to one biological copy with waters and cryoprotectants removed. They are
   NOT F-Fold predictions — F-Fold does not produce usable folds. */

var SCAFFOLD = '#94a3b8';   /* slate  - the scaffold protein in each complex */
var PARTNER  = '#c2410c';   /* orange - site accent, the primary partner     */
var THIRD    = '#2563eb';   /* blue   - the third chain                      */

var HILITE = '#7c3aed';     /* violet - armadillo repeat 1, the BCL9 site */

/* Armadillo repeat 1 (residues 141-180) is the segment BCL9 engages
   (Zhu et al., J Med Chem 2024). Highlighted on the apo structures so the
   druggable site is visible without a partner bound. */
var AR1 = {resi: '141-180', name: 'armadillo repeat 1 (BCL9 site)', color: HILITE};

var STRUCTURES = [
  {
    id: '8Z61', res: '2.5 Å', group: "This lab's depositions",
    label: 'β-catenin, apo (form 1)',
    stage: 'deposited by this group — J Med Chem 2024',
    blurb: 'One of two previously unreported apo β-catenin structures ' +
           'determined by this group. Comparing them against BCL9-bound ' +
           'β-catenin exposed conformational dynamics at the N-terminal ' +
           'armadillo repeats — the movement that governs how inhibitors ' +
           'bind the BCL9 site.',
    cite: 'Zhu et al., J Med Chem (2024)', pmid: '38912577',
    chains: [{c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD}],
    hilite: AR1
  },
  {
    id: '8Z5J', res: '2.8 Å', group: "This lab's depositions",
    label: 'β-catenin, apo (form 2)',
    stage: 'deposited by this group — J Med Chem 2024',
    blurb: 'The second apo form. Superposition of the four N-terminal ' +
           'α-helices across 22 β-catenin crystal structures gave RMSDs from ' +
           '0.34 to 1.73 Å; this form and 8Z61 sit at the divergent end, which ' +
           'is why inhibitor binding modes shift between them.',
    cite: 'Zhu et al., J Med Chem (2024)', pmid: '38912577',
    chains: [{c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD}],
    hilite: AR1
  },
  {
    id: '2GL7', res: '2.6 Å', group: 'Partner complexes',
    label: 'β-catenin / BCL9 / TCF7L2',
    stage: 'nucleus — the ternary transcriptional complex',
    blurb: 'The first reported β-catenin/BCL9/TCF4 structure. BCL9 and TCF7L2 ' +
           'bind non-overlapping faces of the armadillo superhelix, which is ' +
           'what makes the BCL9 groove selectively druggable without ' +
           'disturbing TCF binding.',
    cite: 'Sampietro et al., Mol Cell (2006)', pmid: '17052462',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD},
      {c: 'B', name: 'TCF7L2 / TCF4',      color: PARTNER},
      {c: 'C', name: 'BCL9',               color: THIRD}
    ]
  },
  {
    id: '3SL9', res: '2.2 Å', group: 'Partner complexes',
    label: 'β-catenin / BCL9',
    stage: 'nucleus — the druggable groove',
    blurb: 'BCL9 bound to β-catenin without TCF. The accompanying study ' +
           'identified a labile α-helix abutting the BCL9 site as the element ' +
           'required for inhibition by carnosic acid — chemical evidence that ' +
           'this groove is addressable.',
    cite: 'de la Roche et al., Nat Commun (2012)', pmid: '22353711',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD},
      {c: 'C', name: 'BCL9',               color: PARTNER}
    ]
  },
  {
    id: '1JDH', res: '1.9 Å', group: 'Partner complexes',
    label: 'β-catenin / TCF4',
    stage: 'nucleus — the transcription-factor interface',
    blurb: 'The highest-resolution structure in this set. TCF4 runs along the ' +
           'armadillo groove over a far longer footprint than BCL9 does — the ' +
           'reason TCF-site inhibitors have been harder to make selective.',
    cite: 'Graham et al., Nat Struct Biol (2001)', pmid: '11713475',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD},
      {c: 'B', name: 'TCF4 (hTcf-4)',      color: PARTNER}
    ]
  },
  {
    id: '1M1E', res: '2.1 Å', group: 'Partner complexes',
    label: 'β-catenin / ICAT',
    stage: 'nucleus — the endogenous inhibitor',
    blurb: 'ICAT is the cell\u2019s own β-catenin inhibitor, occluding the ' +
           'armadillo repeats that TCF needs. A natural template for what a ' +
           'protein-protein-interaction blocker has to achieve.',
    cite: 'Daniels & Weis, Mol Cell (2002)', pmid: '12408825',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD},
      {c: 'B', name: 'ICAT (CTNNBIP1)',    color: PARTNER}
    ]
  },
  {
    id: '1I7W', res: '2.0 Å', group: 'Partner complexes',
    label: 'β-catenin / phospho-E-cadherin',
    stage: 'membrane — the adhesion pool',
    blurb: 'The same armadillo groove, used for cell adhesion instead of ' +
           'transcription. Any nuclear-partner inhibitor must avoid this ' +
           'complex — the selectivity problem the medicinal chemistry has to ' +
           'solve.',
    cite: 'Huber & Weis, Cell (2001)', pmid: '11348595',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)',         color: SCAFFOLD},
      {c: 'B', name: 'E-cadherin (phosphorylated)', color: PARTNER}
    ]
  },
  {
    id: '4DJS', res: '3.0 Å', group: 'Inhibitor-bound',
    label: 'β-catenin / stapled peptide',
    stage: 'chemical biology — a designed blocker',
    blurb: 'A hydrocarbon-stapled peptide bound in the armadillo groove: ' +
           'proof that a synthetic ligand can engage this interface, and the ' +
           'starting point for the small-molecule campaigns below.',
    cite: 'Grossmann et al., PNAS (2012)', pmid: '23071338',
    chains: [
      {c: 'A', name: 'β-catenin (CTNNB1)', color: SCAFFOLD},
      {c: 'B', name: 'stapled peptide',    color: PARTNER}
    ]
  },
  {
    id: '4NM7', res: '2.3 Å', group: 'Upstream pathway',
    label: 'GSK-3β / Axin-1 / phospho-LRP6',
    stage: 'cytoplasm — the signalosome switch',
    blurb: 'How Wnt turns the destruction complex off: the phosphorylated ' +
           'LRP6 e-motif occupies the GSK-3β substrate site, with Axin-1 ' +
           'tethered alongside.',
    cite: 'Stamos et al., eLife (2014)', pmid: '24642411',
    chains: [
      {c: 'A', name: 'GSK-3β',                 color: SCAFFOLD},
      {c: 'B', name: 'Axin-1',                 color: PARTNER},
      {c: 'C', name: 'LRP6 e-motif (phospho)', color: THIRD}
    ]
  },
  {
    id: '3S8V', res: '3.1 Å', group: 'Upstream pathway',
    label: 'LRP6 / DKK1',
    stage: 'membrane — receptor antagonism',
    blurb: 'The antagonist DKK1 bound to the LRP6 β-propeller, blocking ' +
           'co-receptor engagement at the top of the pathway.',
    cite: 'Cheng et al., Nat Struct Mol Biol (2011)', pmid: '21984209',
    chains: [
      {c: 'A', name: 'LRP6 (E3E4 propellers)', color: SCAFFOLD},
      {c: 'X', name: 'DKK1',                   color: PARTNER}
    ]
  }
];

var viewer = null, current = -1, spinning = true, cache = {};

function setStatus(msg, isError) {
  var el = document.getElementById('viewer-status');
  if (!el) return;
  el.textContent = msg || '';
  el.style.display = msg ? 'block' : 'none';
  el.className = 'viewer-status' + (isError ? ' error' : '');
}

function renderMeta(s) {
  var key = document.getElementById('viewer-key');
  key.innerHTML = '';
  s.chains.forEach(function (ch) {
    var d = document.createElement('span');
    d.className = 'key-item';
    d.innerHTML = '<i style="background:' + ch.color + '"></i>' + ch.name;
    key.appendChild(d);
  });
  if (s.hilite) {
    var h = document.createElement('span');
    h.className = 'key-item';
    h.innerHTML = '<i style="background:' + s.hilite.color + '"></i>' +
                  s.hilite.name;
    key.appendChild(h);
  }
  document.getElementById('viewer-stage').textContent = s.stage;
  document.getElementById('viewer-blurb').textContent = s.blurb;
  var cite = 'PDB <a href="https://www.rcsb.org/structure/' + s.id + '">' +
             s.id + '</a> · ' + s.res + ' · ' + s.cite;
  if (s.pmid) {
    cite += ' · <a href="https://pubmed.ncbi.nlm.nih.gov/' + s.pmid + '/">PMID ' +
            s.pmid + '</a>';
  }
  document.getElementById('viewer-cite').innerHTML = cite;
  Array.prototype.forEach.call(
    document.querySelectorAll('#viewer-tabs button'), function (b, i) {
      b.setAttribute('aria-selected', i === current ? 'true' : 'false');
    });
}

function paint(s, pdbText) {
  viewer.clear();
  viewer.addModel(pdbText, 'pdb');
  viewer.setStyle({}, {cartoon: {color: SCAFFOLD, opacity: 0.95}});
  s.chains.forEach(function (ch) {
    viewer.setStyle({chain: ch.c},
      {cartoon: {color: ch.color, thickness: 0.9, arrows: true}});
  });
  // the short partner peptides read better with their backbone emphasised
  s.chains.slice(1).forEach(function (ch) {
    viewer.addStyle({chain: ch.c},
      {stick: {radius: 0.12, color: ch.color, opacity: 0.55}});
  });
  if (s.hilite) {
    viewer.setStyle({chain: s.chains[0].c, resi: s.hilite.resi},
      {cartoon: {color: s.hilite.color, thickness: 1.0, arrows: true}});
  }
  viewer.zoomTo();
  viewer.zoom(1.15);
  viewer.spin(spinning ? 'y' : false, 0.45);
  viewer.render();
}

function show(i) {
  if (!viewer || i === current) return;
  var s = STRUCTURES[i];
  current = i;
  renderMeta(s);
  if (cache[s.id]) { paint(s, cache[s.id]); setStatus(''); return; }
  setStatus('Loading ' + s.id + '…');
  fetch('assets/structures/' + s.id + '.pdb')
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    })
    .then(function (t) { cache[s.id] = t; paint(s, t); setStatus(''); })
    .catch(function (e) {
      setStatus('Could not load ' + s.id + ' (' + e.message +
                '). View it at rcsb.org/structure/' + s.id, true);
    });
}

function initViewer() {
  var host = document.getElementById('viewer');
  if (!host) return;
  if (typeof $3Dmol === 'undefined') {
    setStatus('3D viewer failed to load. The structures are at rcsb.org — ' +
              '2GL7, 3SL9, 4NM7, 3S8V.', true);
    return;
  }
  viewer = $3Dmol.createViewer(host, {backgroundColor: '#ffffff'});

  var tabs = document.getElementById('viewer-tabs');
  var seen = {};
  STRUCTURES.forEach(function (s, i) {
    if (!seen[s.group]) {
      seen[s.group] = true;
      var h = document.createElement('div');
      h.className = 'tab-group';
      h.textContent = s.group;
      tabs.appendChild(h);
    }
    var b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'tab');
    b.innerHTML = '<strong>' + s.id + '</strong><span>' + s.label + '</span>';
    b.onclick = function () { show(i); };
    tabs.appendChild(b);
  });

  var sp = document.getElementById('viewer-spin');
  sp.onclick = function () {
    spinning = !spinning;
    sp.textContent = spinning ? 'Pause rotation' : 'Resume rotation';
    sp.setAttribute('aria-pressed', spinning ? 'true' : 'false');
    if (viewer) { viewer.spin(spinning ? 'y' : false, 0.45); }
  };

  // respect users who asked for reduced motion
  var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  if (rm && rm.matches) { spinning = false; sp.textContent = 'Resume rotation'; }

  show(0);
  window.addEventListener('resize', function () {
    if (viewer) { viewer.resize(); viewer.render(); }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initViewer);
} else {
  initViewer();
}
