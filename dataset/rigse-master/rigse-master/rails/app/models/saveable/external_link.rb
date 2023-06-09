class Saveable::ExternalLink < ApplicationRecord
  self.table_name = "saveable_external_links"

  belongs_to :learner,     :class_name => 'Portal::Learner'
  belongs_to :offering,    :class_name => 'Portal::Offering'

  belongs_to :embeddable,  :polymorphic => true

  delegate :name, :to => :embeddable

  # External link can be displayed in an iframe in teacher report.
  delegate :display_in_iframe, :to => :embeddable
  delegate :width, :to => :embeddable
  delegate :height, :to => :embeddable
end
