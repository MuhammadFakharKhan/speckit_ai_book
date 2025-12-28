import React from 'react';
import clsx from 'clsx';
import styles from './RoboticsDiagram.module.css';

const FeatureList = [
  {
    title: 'ROS 2 Architecture',
    icon: '🔄',
    description: (
      <>
        Understanding the core concepts of ROS 2 including nodes, topics, services, and actions.
      </>
    ),
  },
  {
    title: 'Isaac Navigation',
    icon: '💡',
    description: (
      <>
        Implementing navigation systems for humanoid robots using Isaac and Nav2.
      </>
    ),
  },
  {
    title: 'Humanoid Control',
    icon: '🤖',
    description: (
      <>
        Designing control systems for humanoid robot models using URDF and controllers.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <div className={styles.featureIcon}>{icon}</div>
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function RoboticsFeatures() {
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