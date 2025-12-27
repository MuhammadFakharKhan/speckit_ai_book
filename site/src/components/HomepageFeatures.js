import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'ROS 2 Fundamentals',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Learn the core concepts of ROS 2 including nodes, topics, services, and actions
        specifically for humanoid robotics applications.
      </>
    ),
  },
  {
    title: 'Python Agent Integration',
    Svg: require('../../static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Design Python AI agents that interface with ROS 2 using rclpy to bridge
        AI algorithms with robot controllers.
      </>
    ),
  },
  {
    title: 'URDF for Humanoids',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Create and visualize humanoid robot models using Unified Robot Description Format
        with practical examples and exercises.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}