import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

import './styles.module.css';

const FeatureList = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/hands.png').default,
    description: (
      <>
        IAflow was designed from the ground up to be easily installed and
        used to get your workflow up and running quickly.
      </>
    ),
  },
  {
    title: 'Focus on What Matters',
    Svg: require('@site/static/img/target.png').default,
    description: (
      <>
        IAflow lets you focus on your training, and not how to do it.
      </>
    ),
  },
  {
    title: 'Receive Notifications',
    Svg: require('@site/static/img/notification.png').default,
    description: (
      <>
        IAflow allow you send notifications to Discord channels or Telegram message, coming soon WhatsApp messages.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={Svg} className={styles.featureSvg} role="img"/>
      </div>
      <div
        style={{ paddingTop: '2rem' }}
        className="text--center padding-horiz--md">
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
