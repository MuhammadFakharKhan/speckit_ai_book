import React, { memo } from 'react';

const Nav2Icon = memo(({ size = 'md', className, ...props }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const sizeClass = sizes[size] || sizes.md;
  const combinedClassName = `${sizeClass} ${className || ''}`;

  return (
    <svg
      className={combinedClassName}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <circle cx="12" cy="12" r="10" fill="#FF6600" />
      <path d="M12 6l4 8h-8l4-8z" fill="#fff" />
    </svg>
  );
});

export default Nav2Icon;