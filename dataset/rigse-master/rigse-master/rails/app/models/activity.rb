class Activity < ApplicationRecord

  belongs_to :user
  belongs_to :investigation
  belongs_to :original

  has_many :offerings, :dependent => :destroy, :as => :runnable, :class_name => "Portal::Offering"

  has_many :materials_collection_items, :dependent => :destroy, :as => :material
  has_many :materials_collections, :through => :materials_collection_items

  has_many :learner_activities, :dependent => :destroy, :class_name => "Report::LearnerActivity"

  has_many :activity_feedbacks, :class_name => "Portal::LearnerActivity"

  has_many :external_activities, :as => :template

  has_many :sections, -> { order :position }, :dependent => :destroy do
    def student_only
      where('teacher_only' => false)
    end
  end

  has_many :pages, :through => :sections

  has_many :project_materials, :class_name => "Admin::ProjectMaterial", :as => :material, :dependent => :destroy
  has_many :projects, :class_name => "Admin::Project", :through => :project_materials

  has_many :page_elements, :through => :pages

  acts_as_replicatable
  acts_as_list :scope => :investigation

  include Changeable
  include TreeNode
  include Publishable
  include Archiveable
  include Cohorts
  include ResponseTypes
  include HasEmbeddables

  send_update_events_to :investigation

  scope :without_teacher_only, -> {
    where('activities.teacher_only = 0')
  }

  scope :activity_group, -> {
      group("#{self.table_name}.id")
    }

  scope :like, lambda { |name|
    name = "%#{name}%"
    where("#{self.table_name}.name LIKE ? OR #{self.table_name}.description LIKE ?", name, name)
  }

  scope :investigation, -> {
    joins("left outer JOIN investigations ON investigations.id = activities.investigation_id")
  }

  scope :published, -> {
    where('activities.publication_status = "published" OR (investigations.publication_status = "published" AND investigations.allow_activity_assignment = 1)')
  }

  scope :directly_published, -> {
    where('activities.publication_status = "published"')
  }

  scope :assigned, -> { where('offerings_count > 0') }

  scope :ordered_by, lambda { |order| order(order) }

  scope :is_template, ->(v) do
    joins(['LEFT OUTER JOIN investigations ON investigations.id = activities.investigation_id',
           'LEFT OUTER JOIN external_activities',
           'ON (external_activities.template_id = activities.id AND external_activities.template_type = "Activity")',
           'OR (external_activities.template_id = investigations.id AND external_activities.template_type = "Investigation")'])
        .where("external_activities.id IS #{v ? 'NOT' : ''} NULL")
        .distinct
  end
  # End scope weeding zone

  def parent
    return investigation
  end

  def children
    sections
  end

  def left_nav_panel_width
    300
  end

  def deep_xml
    self.to_xml(
      :include => {
        :sections => {
          :exclude => [:id,:activity_id],
          :include => {
            :pages => {
              :exclude => [:id,:section_id],
              :include => {
                :page_elements => {
                  :except => [:id,:page_id],
                  :include => {
                    :embeddable => {
                      :except => [:id,:embeddable_type,:embeddable_id]
                    }
                  }
                }
              }
            }
          }
        }
      }
    )
  end


  # TODO: we have to make this container nuetral,
  # using parent / tree structure (children)
  def reportable_elements
    return @reportable_elements if @reportable_elements
    @reportable_elements = []
    unless teacher_only?
      @reportable_elements = sections.collect{|s| s.reportable_elements }.flatten
      @reportable_elements.each{|elem| elem[:activity] = self}
    end
    return @reportable_elements
  end

  def question_number(embeddable)
    found_index = reportable_elements.find_index { |e| e[:embeddable] == embeddable}
    return -1 unless found_index
    return found_index + 1
  end

  def print_listing
    listing = []
    self.sections.each do |s|
      s.pages.each do |p|
        listing << {"#{s.name} #{p.name}" => p}
      end
    end
    listing
  end

  def full_title
    full_title = self.name
    unless self.parent.nil?
      full_title = "#{full_title} | #{self.parent.name}"
    end

    return full_title
  end

  def is_official
    true # FIXME: Not sure if true should be the hardwired value here
  end

  def is_template
    if (investigation && investigation.external_activities.to_a.compact.length > 0)
      return true
    end
    return external_activities.to_a.compact.length > 0
  end

end
