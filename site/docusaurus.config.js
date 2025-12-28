// @ts-check
// `@type` JSDoc annotations allow IDEs and type-checking tools to autocomplete
//  and provide type information.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ROS 2 for Humanoid Robotics',
  tagline: 'A comprehensive guide to ROS 2 fundamentals, Python agents, and URDF for humanoid robots',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://speckit-ai-book-site.vercel.app/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          path: 'docs',
          routeBasePath: '/docs',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.svg',
      navbar: {
        title: 'ROS 2 for Humanoid Robotics',
        logo: {
          alt: 'ROS 2 Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Tutorial',
          },
          {
            type: 'dropdown',
            label: 'Modules',
            position: 'left',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/ros2-fundamentals',
              },
              {
                label: 'Cognitive Planning',
                to: '/docs/cognitive-planning/',
              },
              {
                label: 'Voice-to-Action',
                to: '/docs/voice-to-action/',
              },
              {
                label: 'VLA Overview',
                to: '/docs/vla-ecosystem-overview',
              },
            ],
          },
          {
            type: 'search',
            position: 'right',
          },
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
            className: 'navbar-github-icon', // CSS class for potential icon styling
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learn',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/ros2-fundamentals',
              },
              {
                label: 'Python Agents',
                to: '/docs/python-agents',
              },
              {
                label: 'URDF for Humanoids',
                to: '/docs/urdf-humanoids',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/ros2',
                className: 'footer-link-stack-overflow',
              },
              {
                label: 'ROS Discourse',
                href: 'https://discourse.ros.org/',
                className: 'footer-link-discord', // Using Discord class for icon
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/ROStwo',
                className: 'footer-link-twitter',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/ros2/ros2',
              },
              {
                label: 'ROS Documentation',
                href: 'https://docs.ros.org/en/rolling/',
              },
              {
                label: 'Robots & Simulation',
                href: 'https://www.ros.org/repositories/',
              },
            ],
          },
        ],
        copyright: `Built with Docusaurus for the ROS 2 Humanoid Robotics Community. Copyright © ${new Date().getFullYear()} ROS 2 for Humanoid Robotics.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;