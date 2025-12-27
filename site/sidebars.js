/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/ros2-fundamentals'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/gazebo-physics',
        'module2/simulated-sensors',
        'module2/unity-integration'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'isaac-ecosystem-overview',
        {
          type: 'category',
          label: 'Isaac Sim & Synthetic Data',
          items: [
            'isaac-sim/index',
            'isaac-sim/photorealistic-simulation',
            'isaac-sim/rtx-rendering-configuration',
            'isaac-sim/usd-scene-composition',
            'isaac-sim/usd-scene-composition-examples',
            'isaac-sim/synthetic-data-generation',
            'isaac-sim/synthetic-dataset-workflows',
            'isaac-sim/domain-randomization'
          ],
        },
        {
          type: 'category',
          label: 'Isaac ROS Perception',
          items: [
            'isaac-ros/index',
            'isaac-ros/hardware-accelerated-perception',
            'isaac-ros/gpu-optimization-techniques',
            'isaac-ros/vslam-pipelines',
            'isaac-ros/perception-pipeline-configurations',
            'isaac-ros/ros2-integration',
            'isaac-ros/sensor-processing-gpu-acceleration'
          ],
        },
        {
          type: 'category',
          label: 'Nav2 for Humanoid Navigation',
          items: [
            'nav2-humanoid/index',
            'nav2-humanoid/path-planning',
            'nav2-humanoid/localization',
            'nav2-humanoid/bipedal-navigation'
          ],
        },
        {
          type: 'category',
          label: 'Integration Examples',
          items: [
            'isaac-sim-to-ros-integration',
            'isaac-ros-to-nav2-integration',
            'isaac-sim-to-nav2-integration',
            'cross-module-references',
            'cross-references'
          ],
        },
        'quickstart',
        'documentation-standards'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) & Autonomous Humanoid',
      items: [
        'vla-overview/index',
        {
          type: 'category',
          label: 'Voice-to-Action Pipeline',
          items: [
            'voice-to-action/index',
            'voice-to-action/speech-recognition-whisper',
            'voice-to-action/intent-parsing',
            'voice-to-action/command-translation',
            'voice-to-action/confidence-scoring',
            'voice-to-action/voice-command-data-model',
            'voice-to-action/stt-processing-workflows',
            'voice-to-action/simulated-voice-examples'
          ],
        },
        {
          type: 'category',
          label: 'Cognitive Planning',
          items: [
            'cognitive-planning/index',
            'cognitive-planning/llm-integration',
            'cognitive-planning/task-decomposition',
            'cognitive-planning/action-sequencing',
            'cognitive-planning/context-awareness',
            'cognitive-planning/data-model',
            'cognitive-planning/validation'
          ],
        },
        {
          type: 'category',
          label: 'Capstone System',
          items: [
            'capstone-system/index',
            'capstone-system/pipeline-integration',
            'capstone-system/simulation-setup',
            'capstone-system/complete-workflow'
          ],
        }
      ],
    },
  ],
};

module.exports = sidebars;