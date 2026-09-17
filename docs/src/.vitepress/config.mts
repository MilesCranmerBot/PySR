import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

// Maps the juliapkg.json pin to the SymbolicRegression.jl docs folder covering
// every version the pin admits, following Julia Pkg compat semantics:
// `=X.Y.Z` is exact, `~X.Y.Z` allows patch bumps, `^X.Y.Z` (or bare) allows minor bumps.
export function symbolicRegressionDocsSubfolder(pkg: { version?: unknown; rev?: unknown }): string {
  const compat = typeof pkg.version === 'string' ? pkg.version.trim().match(/^([=~^]?)(\d+)(?:\.(\d+))?(?:\.(\d+))?$/) : null
  if (compat) {
    const [, op, major, minor, patch] = compat
    if (op === '=' && patch !== undefined) return `v${major}.${minor}.${patch}`
    if ((op === '~' || op === '=' || major === '0') && minor !== undefined) return `v${major}.${minor}`
    return `v${major}`
  }
  return typeof pkg.rev === 'string' && /^v\d+\.\d+\.\d+$/.test(pkg.rev) ? pkg.rev : 'dev'
}

const configDirectory = path.dirname(fileURLToPath(import.meta.url))
const juliaPkg = JSON.parse(readFileSync(path.resolve(configDirectory, '../../../pysr/juliapkg.json'), 'utf8'))
const symbolicRegressionDocsVersion = symbolicRegressionDocsSubfolder(juliaPkg.packages.SymbolicRegression)

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

// Canonical domain for SEO - will be replaced by deploy.jl at build time
// Empty string means no canonical tag (this site is canonical)
const canonicalDomain = '';

const baseTemp = {
  base: '/pysr/',
}

const nav = [
  { text: 'Home', link: '/' },
  { text: 'Examples', link: '/examples' },
  { text: 'API', link: '/api' },
  {
    text: 'Forum',
    link: 'https://github.com/astroautomata/PySR/discussions'
  },
  {
    text: 'Python',
    items: [
      { text: 'Python', link: '/' },
      { text: 'Julia', link: `https://ai.damtp.cam.ac.uk/symbolicregression/${symbolicRegressionDocsVersion}/` }
    ]
  },
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/pysr/',
  title: 'PySR',
  description: 'High-Performance Symbolic Regression in Python and Julia',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../dist',
  srcExclude: ['**/_*.md'],
  head: [
    ['link', { rel: 'icon', href: `${baseTemp.base}favicon.png` }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  ignoreDeadLinks: true,
  transformHead: ({ pageData }) => {
    if (!canonicalDomain) return [];

    // Construct canonical URL for this page
    const pagePath = pageData.relativePath.replace(/((^|\/)index)?\.md$/, '$2');
    const canonicalUrl = `${canonicalDomain}${pagePath}`;

    return [
      ['link', { rel: 'canonical', href: canonicalUrl }]
    ];
  },
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify(getBaseRepository(baseTemp.base)),
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: 'https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/master/images/pysr-logo.png',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/' },
          {
            text: 'Examples',
            link: '/examples',
            collapsed: true,
            items: [
              { text: 'Getting started', link: '/examples/getting-started' },
              { text: 'Expression specifications', link: '/examples/expression-specifications' },
              { text: 'Objectives and losses', link: '/examples/objectives' },
              { text: 'Physics and units', link: '/examples/physics' },
              { text: 'Search behaviour', link: '/examples/search-behaviour' },
              { text: 'Instrumentation and workflow', link: '/examples/instrumentation' },
              { text: 'Value types', link: '/examples/value-types' },
              { text: 'Beyond numeric values', link: '/examples/beyond-numeric-values' },
            ]
          },
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'API Reference', link: '/api' },
          { text: 'Operators', link: '/operators' },
          { text: 'Options', link: '/options' },
          { text: 'Advanced API', link: '/api-advanced' },
        ]
      },
      {
        text: 'Community',
        items: [
          { text: 'Papers', link: '/papers' },
        ]
      },
      {
        text: 'Advanced',
        items: [
          { text: 'Tuning', link: '/tuning' },
          { text: 'Slurm', link: '/slurm' },
          { text: 'Backend', link: '/backend' },
          { text: 'Migrating from v1', link: '/migration' },
        ]
      }
    ],
    editLink: {
      pattern: 'https://github.com/astroautomata/PySR/edit/master/docs/:path',
      text: 'Edit this page on GitHub'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/astroautomata/PySR' }
    ],
    footer: {
      message: 'Made with <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    },
  }
})
