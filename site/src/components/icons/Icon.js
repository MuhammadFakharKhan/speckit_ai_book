import React, { memo } from 'react';
import clsx from 'clsx';

// Define a mapping of icon names to SVG components for robotics-specific icons
const iconMap = {
  // ROS-related icons
  ros: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <circle cx="12" cy="12" r="10" fill="#3399CC" />
      <path d="M8 8h8v8H8z" fill="#fff" />
    </svg>
  )),
  ros2: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <circle cx="12" cy="12" r="10" fill="#3399CC" />
      <text x="12" y="16" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="bold">2</text>
    </svg>
  )),

  // Isaac-related icons
  isaac: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <rect x="2" y="2" width="20" height="20" rx="3" fill="#00B050" />
      <path d="M8 8h8v8H8z" fill="#fff" />
    </svg>
  )),

  // Nav2-related icons
  nav2: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <circle cx="12" cy="12" r="10" fill="#FF6600" />
      <path d="M12 6l4 8h-8l4-8z" fill="#fff" />
    </svg>
  )),

  // General robotics icons
  robot: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <rect x="6" y="4" width="12" height="16" rx="2" fill="#444" />
      <circle cx="9" cy="9" r="1.5" fill="#fff" />
      <circle cx="15" cy="9" r="1.5" fill="#fff" />
      <rect x="8" y="13" width="8" height="3" rx="1" fill="#666" />
    </svg>
  )),

  // Default icon for fallback
  default: memo(() => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
      <circle cx="12" cy="12" r="10" fill="#ccc" />
      <path d="M9 10l2 2 4-4" stroke="#666" strokeWidth="2" fill="none" />
    </svg>
  ))
};

// Icon component that accepts a name prop and renders the corresponding icon
const Icon = memo(({ name, size = 'md', className, ...props }) => {
  const iconSizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const IconComponent = iconMap[name] || iconMap.default;

  const sizeClass = iconSizeClasses[size] || iconSizeClasses.md;
  const combinedClassName = clsx(sizeClass, className);

  return (
    <span className={combinedClassName} {...props}>
      <IconComponent />
    </span>
  );
});

export default Icon;